from abc import ABC
import pandas as pd
from typing import Dict
import logging
import src.utils as U
import src.constants as CONST
import src.config as CONF
from src.models.CNN_simple import CNNSimple
from src.models.CNN_global_max_pooling import CNNGlobalMaxPooling
import torch
from src.core import BaseTransformer
from src.train_test_split import split_df

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger('pipeline-log')
log.setLevel(logging.DEBUG)


class Pipeline(ABC):
    def __init__(self,
                 preprocessor: BaseTransformer,
                 augmenter: BaseTransformer,
                 convertor: BaseTransformer,
                 model_name: str,
                 splitting_params: dict = None,
                 ):
        """
        Pipeline to unite data processing, model fitting and predictions and counting scores for it's performance

        :param preprocessor: basic preprocessor instance
        :param model: custom model with necessary methods: fit, set_catalog_item_train_columns, get_recommendations
        :param splitting_params: dict of parameters for splitting (see split_df)
        """
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.convertor = convertor
        self.model_name = model_name
        self.splitting_params = splitting_params
        self.model = None
        self.optimizer = None

    def run(
            self,
            df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Run pipeline
        :param df: dataframe
        :return recommendations: dictionary of top N catalog items for each user_id
        :return scores: scores for model performance
        """
        # df = df.copy()
        # log.info('Running pipeline')
        # X_train, y_train, X_test, y_test = self.prepare_data(df)
        # self.train_model(X_train, y_train)
        # y_pred = self.get_prediction(X_test)
        #
        # log.info('Counting scores')
        # scores = self.get_scores(np.array(y_test), np.array(y_pred))
        # return scores
        pass

    def train_model(self,
                    learning_rate,
                    criterion: object,
                    optimizer: object,
                    train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    batch_size=CONST.BATCH_SIZE,
                    epochs=CONST.EPOCHS,
                    iter_per_validation=100,
                    early_stopping=False,
                    checkpoint_path="./best_checkpoint",
                    device=CONF.DEVICE
                    ):
        """
        Fit model
        :param X_train:
        :param y_train:
        :return:
        """
        log.info('Fitting model')
        model = None
        if self.model_name == 'CNNSimple':
            model = CNNSimple(n_tokens=len(self.convertor.get_tokens())).to(CONF.DEVICE)
        elif self.model_name == 'CNNGlobalMaxPooling':
            model = CNNGlobalMaxPooling(n_tokens=len(self.convertor.get_tokens())).to(CONF.DEVICE)

        optimizer = optimizer(model.parameters(), lr=learning_rate)

        self.model = model
        self.optimizer = optimizer

        iteration_list, loss_list, accuracy_list = U.train(self.convertor, model,
                                                           optimizer, criterion,
                                                           train_df, val_df,
                                                           batch_size=batch_size,
                                                           epochs=epochs,
                                                           iter_per_validation=iter_per_validation,
                                                           early_stopping=early_stopping,
                                                           checkpoint_path=checkpoint_path,
                                                           device=device)

        return iteration_list, loss_list, accuracy_list

    def draw_curves(self, iteration_list,
                    loss_list, accuracy_list):
        U.draw_visualization(iteration_list, loss_list, accuracy_list)

    def predict(self,
                X: pd.DataFrame,
                checkpoint_path='./best_checkpoint'
                ):
        log.info('Starting prediction')
        _ = U.load_checkpoint(checkpoint_path, self.model, self.optimizer)

        preds = []

        with torch.no_grad():
            for batch in U.iterate_minibatches(self.convertor, X,
                                               batch_size=128, shuffle=False,
                                               device=CONF.DEVICE):  # batch_size=30766
                test_outputs = self.model(batch)
                predicted = torch.max(test_outputs.data, 1)[1]
                preds.extend(predicted)

        return preds

    def get_scores(self,
                   y_true,
                   y_pred
                   ):
        U.calc_metrics(y_true, y_pred)

    def prepare_data(self,
                     df: pd.DataFrame,
                     ):
        """
        Prepare data for fitting and prediction
        :param df: Dataframe to prepare
        :return: four dataframes X_train, y_train, X_test, y_test
        """
        df = df.copy()

        log.debug('Starting preprocessing')
        self.preprocessor.fit(df)
        df = self.preprocessor.transform(df)

        log.debug('Splitting dataset on train, val and test')
        train_df, test_df = split_df(df, **self.splitting_params)
        train_df, val_df = split_df(train_df, **self.splitting_params)

        if self.augmenter is not None:
            log.debug('Augment text for train')
            train_df = self.augmenter.transform(train_df)

        val_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_df.dropna(inplace=True)
        val_df.dropna(inplace=True)
        test_df.dropna(inplace=True)

        data = pd.concat([train_df.assign(indic="train"),
                          val_df.assign(indic="val"),
                          test_df.assign(indic="test")])

        log.debug('Make vocabulary')
        self.convertor.fit(data)

        log.debug('Split data to train/val/test after making vocabulary')
        train_df, val_df, test_df = data[data["indic"].eq("train")], data[data["indic"].eq("val")], data[
            data["indic"].eq("test")]
        return train_df, val_df, test_df
