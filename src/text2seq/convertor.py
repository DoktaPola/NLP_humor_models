import pandas as pd
import src.schema as S
from src.core import BaseTransformer
import numpy as np
from collections import Counter
import torch
import nltk


class Text2SeqConvertor(BaseTransformer):
    """

    """

    def __init__(self):
        super().__init__()

        self.UNK = "UNK"
        self.PAD = "PAD"
        self.min_count = 10
        self.vocab_size = 0
        self.tokens = None
        self.token_to_id = None
        self.UNK_IX = None
        self.PAD_IX = None

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
        tokenizer = nltk.tokenize.WordPunctTokenizer()
        X[S.JOKE] = X[[S.JOKE]].applymap(lambda x: " ".join(tokenizer.tokenize(x.lower())))

        token_counts = Counter()
        for line in X[S.JOKE].values:
            token_counts.update(line.split(" "))

        tokens = sorted(t for t, c in token_counts.items() if c >= self.min_count)
        tokens = [self.UNK, self.PAD] + tokens

        self.tokens = tokens
        self.vocab_size = len(tokens)
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.UNK_IX, self.PAD_IX = map(self.token_to_id.get, [self.UNK, self.PAD])

    def get_tokens(self):
        return self.tokens

    def get_vocab_size(self):
        return self.vocab_size

    def get_token_to_id(self):
        return self.token_to_id

    def get_unk_pad_ix(self):
        return self.UNK_IX, self.PAD_IX

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
        pass
