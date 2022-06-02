import pandas as pd
from typing import Optional
from src.core import BaseTransformer
import src.schema as S
import src.config as CONF
import src.utils as U


class SimplePreprocessor(BaseTransformer):
    """
    Transformer class for short and simple requests dataset preprocessing.

    Returns
    -------
    X : pd.DataFrame
        Preprocessed dataframe.
    """

    def __init__(self, full_joke=True):
        self.full_joke = full_joke

    def _process_missing_values(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Process missing values.
        """
        df = df.copy()
        for col in df.columns:
            if col not in [S.TITLE, S.SCORE]:
                df[col] = df[col].fillna(CONF.VALUE_TO_FILL_NA)
        return df

    def _drop_missing(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Drop missing values.
        """
        df.dropna(subset=[S.TITLE, S.SCORE], inplace=True)
        return df

    def _select_required_columns(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Choose just necessary columns.
        """
        for column in S.SHORT_PREPROCESSING_REQUIRED_COLUMNS:
            assert column in df.columns, f'Missing {column} column'
        df = df.loc[:, S.SHORT_PREPROCESSING_REQUIRED_COLUMNS]
        return df

    def _transform_text(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        # make full joke
        if self.full_joke:
            df[S.JOKE] = df[S.TITLE] + " " + df[S.BODY]
        else:
            df[S.JOKE] = df[[S.TITLE]]
        df.drop([S.TITLE, S.BODY], axis=1, inplace=True)

        # remove html tags
        df[S.JOKE] = df[S.JOKE].apply(U.strip_html_tags)

        # remove extra whitespaces
        df[S.JOKE] = df[S.JOKE].apply(U.remove_whitespace)

        # remove accented characters
        df[S.JOKE] = df[S.JOKE].apply(U.remove_accented_chars)

        # expand contractions
        df[S.JOKE] = df[S.JOKE].apply(U.expand_contractions)

        # remove URLs
        df[S.JOKE] = df[S.JOKE].apply(lambda text: U.remove_urls(text))

        # remove of numbers
        df[S.JOKE] = df[S.JOKE].apply(lambda text: U.remove_numbers(text))

        # drop duplicates
        df.drop_duplicates(inplace=True)
        return df

    def _normalize_target(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        If same jokes have different scores.
        This function set the highest local score for each joke of the group.
        """
        duplicates = df[df[S.JOKE].duplicated() == True][S.JOKE].unique()

        for j in duplicates:
            max_score = max(set(df[df[S.JOKE] == j][S.SCORE]))
            joke_idxs = df[df[S.JOKE] == j].index
            df[S.SCORE][joke_idxs] = max_score

        df.drop_duplicates(inplace=True)
        return df

    def __add_rank(self, data):
        if data == 0:
            return 0
        elif (data > 0) and (data <= 1):
            return 1
        elif (data > 1) and (data <= 5):
            return 2
        elif (data > 5) and (data <= 26):
            return 3
        elif data > 26:
            return 4

    def _transform_target(
            self,
            df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add 5 classes (target):
        0 -> not a joke at all
        1 -> so-so joke
        2 -> normal joke
        3 -> funny joke
        4 -> the funniest joke
        """
        df[S.TARGET] = df[S.SCORE].apply(self.__add_rank)

        # drop joke from one word:
        df[S.TXT_WORD_CNT] = df[S.JOKE].apply(lambda x: len(str(x).split()))
        df = df[df[S.TXT_WORD_CNT] != 1]
        # df.drop([S.TXT_WORD_CNT, S.SCORE], axis=1, inplace=True)
        df.drop([S.SCORE], axis=1, inplace=True)
        return df

    def _fit_df(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None
    ) -> None:
        pass

    def _transform_df(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        X = self._drop_missing(X)
        X = self._process_missing_values(X)
        X = self._transform_text(X)
        X = self._normalize_target(X)
        X = self._transform_target(X)
        return X
