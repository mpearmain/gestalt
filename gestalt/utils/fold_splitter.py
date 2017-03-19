"""
The :mod:`gestalt.utils.fold_splitter` module includes classes and functions to subset data splits based on a
fold strategy from sklearn.model_selection e.g. Kfold.
"""
import pandas as pd
import numpy as np


def split_folds(train_index, test_index, X, y = None):
    # split for pandas data
    if isinstance(X, pd.DataFrame):
        X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
    if y is not None and isinstance(y, pd.DataFrame):
        y_train, y_test = y.ix[train_index, 0].values, y.ix[test_index, 0].values

    if not isinstance(X, pd.DataFrame):
        # This is really hacky but covers numpy and csr sparse
        X_train, X_test = X[train_index], X[test_index]
    if y is not None and isinstance(y, np.ndarray):
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test