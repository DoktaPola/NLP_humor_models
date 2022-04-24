from abc import ABC
import numpy as np
import pandas as pd
from typing import List, Dict
import logging

from src.var_typing import ColumnName
import src.schema as S
from src.core import BaseTransformer
from src.train_test_split import split_df
import config as CONF

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger('pipeline-log')
log.setLevel(logging.DEBUG)


class Pipeline(ABC):
    def __init__(self,
                 preprocessor: BaseTransformer,
                 augmenter: BaseTransformer,
                 model: object,
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
        self.model = model
        self.splitting_params = splitting_params

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
        df = df.copy()
        log.info('Running pipeline')
        X_train, y_train, X_test, y_test = self.prepare_data(df)
        self.fit_model(X_train, y_train)
        y_pred = self.get_recommendations(X_test)

        log.info('Counting scores')
        scores = self.get_scores(np.array(y_test), np.array(y_pred))
        return scores

    def get_scores(self,
                   y_true: np.array,
                   y_pred: np.array
                   ) -> dict:
        return dict([(score_name, score_func(y_true, y_pred)) for score_name, score_func in CONF.SCORES_MAP.items()])

    def fit_model(self,
                  X_train: pd.DataFrame,
                  y_train: pd.DataFrame,
                  ):
        """
        Fit model
        :param X_train:
        :param y_train:
        :return:
        """
        log.info('Fitting model')
        self.model.fit(X_train, y_train)

    def _predict_probabilities(self,
                               X: pd.DataFrame
                               ) -> list:
        # log.info('Starting prediction')
        pass

    def get_recommendations(self,
                            X_test: pd.DataFrame
                            ) -> pd.DataFrame:
        """
        Get recommendations
        :param X_test:
        :return:
        """
        # log.info('accumulating recommendations')
        pass

    def cut_rare_target(self,
                        df: pd.DataFrame,
                        quantile: float,
                        ) -> pd.DataFrame:
        """Func to drop rare target values"""
        df = df.copy()
        items_freq_relative = df[S.TARGET].value_counts().cumsum() / len(df)
        target_values_to_keep = items_freq_relative[items_freq_relative < quantile].index
        df = df[df[S.TARGET].isin(target_values_to_keep)]
        return df

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

        log.debug('Splitting dataset on train and test')
        train_df, test_df = split_df(df, **self.splitting_params)
        if (self.target_quantile is not None) and (self.target_quantile < 1):
            log.debug(f'Cutting {self.target_quantile} target quantile')
            train_df = self.cut_rare_target(train_df, self.target_quantile)

        log.debug('Generating X and y for train and test')
        train_df = self.df_generator.transform(train_df)
        test_df = self.df_generator.transform(test_df)

        log.debug('Separating train and test to X and y')
        X_train = train_df.drop(['y'], axis=1)
        y_train = train_df['y']

        X_test = test_df.drop(['y'], axis=1)
        y_test = test_df['y']

        log.debug('Getting y for test df')
        return X_train, y_train, X_test, y_test

    def prepare_all_train(self,
                          df: pd.DataFrame):
        """
        Prepare data for fitting and prediction
        :param df: Dataframe to prepare
        :return: four dataframes X_train, y_train, X_test, y_test
        """
        df = df.copy()
        log.debug('Preparing whole dataset for training model')
        log.debug('Starting transforming')
        df = self.preprocessor.transform(df)

        if (self.target_quantile is not None) and (self.target_quantile < 1):
            log.debug(f'Cutting {self.target_quantile} target quantile')
            df = self.cut_rare_target(df, self.target_quantile)

        log.debug('Generating X and y for dataset')
        df = self.df_generator.transform(df)

        log.debug('Separating X and y')
        X_train = df.drop(['y'], axis=1)
        y_train = df['y']

        return X_train, y_train

    def predict_items(self,
                      X: pd.DataFrame,
                      probs: np.ndarray,
                      group_by_cols: List[ColumnName] = [S.USER_ID],
                      top_k: int = CONF.N_REC_PROACTIVE,
                      y_prefix: str = CONF.DEFAULT_REC_PREFIX,
                      ) -> pd.DataFrame:
        """
        Transforms predicted probabilities into recommendations of a given length.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data being predicted.

        probs : array-like of shape (n_samples, 2)
            Predicted probabilities (predict_proba result).

        group_by_cols : array_like, default=['USER_ID']
            List of columns defining user type.

        top_k : int, default=4
            Number of created recommendations per user.

        y_prefix: str, default='top_'
            Prefix for columns in resulting dataframe.

        Returns
        -------
        recs : array-like of shape (n_users, top_k)
            Dataframe with recommendations for each user.
        """
        X = X.copy()
        assert len(group_by_cols) == 1, 'It works only with one column for grouping for now'
        group_by_col = group_by_cols[0]
        X['y_pred_proba'] = probs[:, 1]
        if S.TARGET not in X.columns:
            encoded_target = X[[col for col in X.columns if S.TARGET in col]]
            X[S.TARGET] = [enc_val.replace(S.TARGET + '_', '') for enc_val in encoded_target.idxmax(1)]

        X_by_user = pd.DataFrame(X.groupby(group_by_col)[S.TARGET].apply(np.array))
        X_by_user['pred_probas'] = X.groupby(group_by_col)['y_pred_proba'].apply(np.array)
        X_by_user['pred_items'] = X_by_user.apply(
            lambda row: row[S.TARGET][np.array(-row['pred_probas']).argsort()][:top_k], axis=1)

        predicted_items_df = pd.DataFrame(
            np.array([list(preds) for preds in list(X_by_user['pred_items'])]),
            index=np.array(X_by_user.index),
            columns=[y_prefix + str(i) for i in range(1, top_k + 1)])

        predicted_items_df.index.rename(group_by_col, inplace=True)

        return predicted_items_df
