"""
The :mod:`gestalt.utils.fold_splitter` module includes classes and functions to subset data splits based on a
fold strategy from sklearn.model_selection e.g. Kfold.
"""

def split_pandas(train_index, test_index, X, y = None):
    X_train, X_test = X.ix[train_index, :], X.ix[test_index, :]
    if y is not None:
        y_train, y_test = y.ix[train_index, 0].values, y.ix[test_index, 0].values

    return X_train, X_test, y_train, y_test


def split_numpy(train_index, test_index, X, y=None):
    X_train, X_test = X[train_index], X[test_index]
    if y is not None:
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test