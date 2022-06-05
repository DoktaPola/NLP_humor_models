import pandas as pd
import src.schema as S
from src.core import BaseTransformer
import numpy as np


class Augmenter(BaseTransformer):
    """
    Class for text augmentation.
    """

    def __init__(self, augemtation_obj: object):
        super().__init__(augemtation_obj=augemtation_obj)

        self.aug = augemtation_obj

    def _fit_df(self, X: pd.DataFrame, y=None):
        pass

    def __set_aug(self, txt):
        """
        Set different augmenters to input text.
        :param txt: input text
        :return: augmented text
        """
        return self.aug.augment(txt)

    def _transform_df(self,
                      X: pd.DataFrame
                      ) -> pd.DataFrame:
        """
        Transform X using augmentation.

        Parameters
        ----------
        X : Whole dataframe.

        Returns
        -------
        X (DataFrame): dataset with augmented columns
        """
        X_ = X.copy(deep=True)

        X_[S.TXT_AUGMENT] = X_[S.JOKE].apply(self.__set_aug)

        X_.drop([S.JOKE], axis=1, inplace=True)
        X_.rename({S.TXT_AUGMENT: S.JOKE}, axis=1, inplace=True)

        full_df = X.append(X_, ignore_index=True, sort=False)

        # shuffle
        full_df = full_df.reindex(np.random.permutation(full_df.index)).reset_index(drop=True)
        return full_df
