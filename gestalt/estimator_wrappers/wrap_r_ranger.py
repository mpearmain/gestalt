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

    We pull in all the options that ranger allows, but as this is a classifier we hard-code
    probability=True, to give probability values.
    """

    def __init__(self, formula='RANGER_TARGET_DUMMY~.', num_trees=500, num_threads=1, verbose=True, seed=42):
        self.formula = formula
        self.num_trees = num_trees
        self.probability = True
        self.num_threads = num_threads
        self.verbose = verbose
        self.seed = seed
        self.num_classes = None
        self.clf = None

    def fit(self, X, y):
        # First convert the X and y into a dataframe object to use in R
        # We have to convert the y back to a dataframe to join for using with R
        # We give the a meaningless name to allow the formula to work correctly.
        y = pd.DataFrame(y, index=X.index, columns = ['RANGER_TARGET_DUMMY'])
        self.num_classes = y.ix[:, 0].nunique()
        r_dataframe = pd.concat([X, y], axis=1)
        r_dataframe['RANGER_TARGET_DUMMY'] = r_dataframe['RANGER_TARGET_DUMMY'].astype('str')
        self.clf = ranger.ranger(formula=self.formula,
                                 data=r_dataframe,
                                 num_trees=self.num_trees,
                                 probability=self.probability,
                                 num_threads=self.num_threads,
                                 verbose=self.verbose,
                                 seed=self.seed)
        return

    def predict_proba(self, X):
        # Ranger doesnt have a specific separate predict and predict probabilities class, it is set in the params
        # REM: R is not Python :)
        pr = R.predict(self.clf, dat=X)
        pandas_preds = ro.pandas2ri.ri2py_dataframe(pr.rx('predictions')[0])
        if self.num_classes == 2:
            pandas_preds = pandas_preds.ix[:, 1]
        return pandas_preds