from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import TransformerMixin


class BaseTransformer(TransformerMixin, ABC):
    def __init__(self, **config):
        self.config = config

    @abstractmethod
    def _fit_df(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        """
        Method for fitting transformer. Should be defined in child classes.
        :param X: input dataframe
        :param y: input target series
        """
        pass

    @abstractmethod
    def _transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Method for transforming dataframe. Should be defined in child classes.
        :param X: input dataframe
        :return:
        """
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Common entry point for fitting transformer.
        :param X: input dataframe
        :param y: input target series
        """
        # we can add here some additional validations and common logic for all transformers
        if y is not None:
            assert X.shape[0] == y.shape[0], "The length of X and y has to be the same."
            y = y.copy()

        X = X.copy()
        self._fit_df(X, y)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Common entry point for transforming dataframe.
        :param X: input dataframe
        :return: transformed dataframe
        """
        # we can add here some additional validations and common logic for all transformers
        X = X.copy()
        return self._transform_df(X)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        """
        Common entry point for fitting and transforming dataframe.
        :param X: input dataframe
        :param y: target column (optional)
        :param fit_params:
        :return:
        """
        self.fit(X, y)
        return self.transform(X)

    def get_config(self) -> Dict[str, Any]:
        """
        Extracting config of created instance
        :return: dict with config
        """
        return self.config
