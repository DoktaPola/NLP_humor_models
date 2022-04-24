import pandas as pd
import src.schema as S
import src.constants as C
from sklearn.model_selection import train_test_split


def split_df(X: pd.DataFrame,
             test_size=None,
             random_state=None,
             shuffle=True,
             # stratify=None
             ):
    """
    Split dataset on train/test by user_id.

    Parameters
    ----------
    X (pd.DataFrame): Initial dataset
    test_size (float or int, default=None): If float, should be between 0.0 and 1.0 and represent the proportion
                                            of the dataset to include in the test split. If int, represents the
                                            absolute number of test samples. If None, the value will
                                            be set to 0.25.

    random_state (int, default=None): Controls the shuffling applied to the data before applying the split.
                                      If None, the value will be set to 42.

    shuffle (bool, default=True): Whether or not to shuffle the data before splitting. If shuffle=False
                                  then stratify must be None.

    stratify (str, default=None): If str (name of column for stratification), data is split in a stratified fashion, using this as
                                         the class labels.

    Returns
    -------
    train and test datasets (pd.DataFrame)
    """

    if not test_size:
        test_size = C.DEFAULT_TEST_SIZE

    if shuffle and random_state is None:
        random_state = C.DEFAULT_RANDOM_STATE

    X_train, X_test = train_test_split(X,
                                       test_size=test_size,
                                       random_state=random_state,
                                       shuffle=shuffle,
                                       # stratify=stratify
                                       )
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    return X_train, X_test
