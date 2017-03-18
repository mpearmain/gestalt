# import sklearn BaseEstimator etc to use
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

# Activate R objects.
pandas2ri.activate()
R = ro.r

# import the R packages required.
base = importr('base')
ranger = importr('ranger')

"""
An example of how to make an R plugin to use with Gestalt, this way we can run our favourite R packages using the
same stacking framework.
"""

class RangerClassifier(BaseEstimator, ClassifierMixin):
    """
    From Ranger DESCRIPTION FILE:
    A fast implementation of Random Forests, particularly suited for high dimensional data.
    Ensembles of classification, regression, survival and probability prediction trees are supported.
    """

    def __init__(self, params={}):
        self.params = params
        self.clf = None

    def fit(self, X, y):
        # First convert the X and y into a dataframe object to use in R
        if X.index != y.index:
            raise Exception('X and y must have the same index.')
        y_colname = list(y)
        r_dataframe = pd.concat([X, y], axis=1)
        robjects.globalenv['dataframe'] = r_dataframe
        self.clf = ranger('y~x', data=base.as_symbol('dataframe'))
        return

    def predict(self, X):
        robjects.globalenv['dataframe'] = X
        clf = self.clf
        preds = predict.ranger(base.as_symbol('dataframe'))

        return preds

    def predict_proba(self, X):
        # Ranger doesnt have a specific separate predict and predict probabilities class, it is set in the params
        # REM: R is not Python :)
        preds = self.predict(X)
        return preds
