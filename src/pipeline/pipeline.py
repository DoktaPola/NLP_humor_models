from abc import ABC
import pandas as pd
from typing import Dict
import logging
import src.utils as U
import src.schema as S
import src.constants as CONST
import src.config as CONF
import torch
from src.core import BaseTransformer
from src.train_test_split import split_df
from src.evaluate.metrics import calc_metrics

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger('pipeline-log')
log.setLevel(logging.DEBUG)


class Pipeline(ABC):
    def __init__(self,
                 preprocessor: BaseTransformer,
                 augmenter: BaseTransformer,
                 convertor: BaseTransformer,
                 model,
                 splitting_params: dict = None,
                 ):
        """
        Pipeline to unite data processing, model training and predictions and counting scores for it's performance

        :param preprocessor: basic preprocessor instance
        :param augmenter: basic augmenter (synonyms imputer)
        :param convertor: custom convertor words to vectors
        :param model: custom model
        :param splitting_params: dict of parameters for splitting (see split_df)
        """
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.convertor = convertor
        self.model = model
        self.splitting_params = splitting_params
        # self.model = None
        self.optimizer = None

    def run_multi_classifier(
            self,
            X: pd.DataFrame,
            learning_rate,
            criterion: object,
            optimizer: object,
            epochs: int = 10,
            checkpoint_path='../checkpoints/best_checkpoint'
            ):
        """
        Run pipeline.
        """
        df = X.copy()
        log.info('Running multi classifier pipeline')
        train_df, val_df, test_df = self.prepare_data(df)
        iteration_list, loss_list, accuracy_list = self.train_model(learning_rate,
                                                                    criterion,
                                                                    optimizer,
                                                                    train_df,
                                                                    val_df,
                                                                    epochs=epochs,
                                                                    checkpoint_path=checkpoint_path)

        self.draw_curves(iteration_list, loss_list, accuracy_list)
        pred = self.predict(test_df, checkpoint_path=checkpoint_path)
        self.get_scores(test_df[S.TARGET].values, pred)

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
        Train model
        :return: data for drawing curves
        """
        log.info('Training model')
        log.info(f'*** {self.model}')
        model = self.model(n_tokens=len(self.convertor.get_tokens())).to(CONF.DEVICE)
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
                    loss_list, accuracy_list
                    ):
        """
        Visualize loss and accuracy of the model
        """
        U.draw_visualization(iteration_list, loss_list, accuracy_list)

    def predict(self,
                X: pd.DataFrame,
                checkpoint_path='./best_checkpoint'
                ):
        """
        Get classes prediction.
        :return: classes
        """
        log.info('Starting prediction')
        _ = U.load_checkpoint(checkpoint_path, self.model, self.optimizer)

        preds = []

        with torch.no_grad():
            for batch in U.iterate_minibatches(self.convertor, X,
                                               batch_size=128, shuffle=False,
                                               device=CONF.DEVICE):
                test_outputs = self.model(batch)
                predicted = torch.max(test_outputs.data, 1)[1]
                preds.extend(predicted)

        return preds

    def get_scores(self,
                   y_true,
                   y_pred
                   ):
        calc_metrics(y_true, y_pred)

    def prepare_data(self,
                     df: pd.DataFrame,
                     ):
        """
        Prepare data for training and prediction
        :param df: Dataframe to prepare
        :return: three dataframes train_df, val_df, test_df
        """
        df = df.copy()

        log.debug('Starting preprocessing')
        self.preprocessor.fit(df)
        df = self.preprocessor.transform(df)
        df.drop([S.TXT_WORD_CNT], axis=1, inplace=True)

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
