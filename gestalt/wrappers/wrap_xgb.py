import numpy as np
import pandas as pd
# Wrapper Class of Classifiers
from gestalt.models.gestalt import Gestalt
# BaseEstimator
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin

import xgboost as xgb

# Although the python version of xgboost does come with a sklearn interface it doesnt contain ALL the params we may want
# to use, these wrapper by pass that need.


class XGBClassifier(BaseEstimator, ClassifierMixin):
    '''
    XGBClassifier in xgboost for sklearn doesnt have ALL parameters accessible, a simple wrapper to expose them

    (Example)
    from models import XGBClassifier
    class XGBModelV1(XGBClassifier):
        def __init__(self,**params):
            super(XGBModelV1, self).__init__(**params)

    a = XGBModelV1(colsample_bytree=0.9,
                   learning_rate=0.01,
                   max_depth=5,
                   min_child_weight=1,
                   n_estimators=300,
                   nthread=-1,
                   objective='binary:logistic',
                   seed=0,
                   silent=True,
                   subsample=0.8)
    a.fit(X_train, y_train, evals=[(X_train, y_train),(X_test, y_test)])

    '''

    def __init__(self, params={}, num_round=50, early_stopping_rounds=None, verbose_eval=True):
        self.params = params
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose_eval
        self.clf = xgb

    def fit(self, X, y, X_test=None, y_test=None):

        dtrain = xgb.DMatrix(X, label=y)
        if X_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test)
            watchlist = [(dtrain, 'train'), (dtest, 'validation')]

        else:
            watchlist = [(dtrain, 'train')]

        self.clf = xgb.train(params=self.params,
                             dtrain=dtrain,
                             num_boost_round=self.num_round,
                             evals=watchlist,
                             early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=self.verbose)
        return self.clf

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = pd.DataFrame(self.clf.predict(dtest), index=X.index)
        return preds


# Regressor Wrapper Class
class XGBRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from models import XGBClassifier
    class XGBModelV1(XGBClassifier):
        def __init__(self,**params):
            super(XGBModelV1, self).__init__(**params)

    a = XGBModelV1(colsample_bytree=0.9,
                   learning_rate=0.01,
                   max_depth=5,
                    min_child_weight=1,
                    n_estimators=300,
                    nthread=-1,
                    objective='reg:linear',
                    seed=0,
                    silent=True,
                    subsample=0.8)
    a.fit(X_train, y_train, eval_metric='rmse',eval_set=[(X_train, y_train),(X_test, y_test)])

    """

    def __init__(self, params={}, num_round=50, eval_metric=None, early_stopping_rounds=None, verbose_eval=True):
        self.params = params
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.verbose = verbose_eval
        self.clf = xgb

    def fit(self, X, y, X_test=None, y_test=None):

        dtrain = xgb.DMatrix(X, label=y)

        if X_test is not None:
            dtest = xgb.DMatrix(X_test, label=y_test)
            watchlist = [(dtrain, 'train'), (dtest, 'validation')]

        else:
            watchlist = [(dtrain, 'train')]

        self.clf = xgb.train(params=self.params,
                             dtrain=dtrain,
                             num_boost_round=self.num_round,
                             evals=watchlist,
                             early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=self.verbose)
        return self.clf

    def predict(self, X):
        dtest = xgb.DMatrix(X)

        return pd.DataFeself.clf.predict(dtest)

