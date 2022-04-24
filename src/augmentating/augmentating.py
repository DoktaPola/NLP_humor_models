# from typing import List
#
import pandas as pd
import src.schema as S
from src.core import BaseTransformer
import numpy as np

class Augmenter(BaseTransformer):
    """

    """

    def __init__(self, augemtation_obj:object):
        super().__init__(augemtation_obj=augemtation_obj)

        self.aug = augemtation_obj

    def _fit_df(self, X: pd.DataFrame, y=None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored.

        Returns
        -------
        self
            Fitted encoder.
        """
        pass

    def __set_aug(self, txt):
        return self.aug.augment(txt)

    def _transform_df(self,
                      X: pd.DataFrame
                      ) -> pd.DataFrame:
        """
        Transform X using one-hot encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X (DataFrame): dataset with encoded columns (original columns - dropped)
        """
        X_ = X.copy(deep=True)

        X_[S.TXT_AUGMENT] = X_[S.JOKE].apply(self.__set_aug)

        X_.drop([S.JOKE], axis=1, inplace=True)
        X_.rename({S.TXT_AUGMENT: S.JOKE}, axis=1, inplace=True)

        full_df = X.append(X_, ignore_index=True, sort=False)

        # shuffle
        full_df = full_df.reindex(np.random.permutation(full_df.index)).reset_index(drop=True)
        return full_df
