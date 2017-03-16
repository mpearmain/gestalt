import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin


# Although the python version of xgboost does come with a sklearn interface it doesnt contain ALL the params we may want
# to use, these wrapper by pass that need.


class XGBClassifier(BaseEstimator, ClassifierMixin):
    """
    XGBClassifier in xgboost for sklearn doesnt have ALL parameters accessible, a simple wrapper to expose them
    params = {colsample_bytree=0.9,
              learning_rate=0.01,
              max_depth=5,
              min_child_weight=1,
              n_estimators=300,
              nthread=-1,
              objective='binary:logistic',
              seed=0,
              silent=True,
              subsample=0.8}
    a = XGBClassifier(params=params)
    a.fit(X_train, y_train)

    """

    def __init__(self, params={}, num_round=50, early_stopping_rounds=None, verbose_eval=True):
        self.params = params
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose_eval
        self.xgb = None

    def fit(self, X, y):

        dtrain = xgb.DMatrix(X, label=y)
        self.xgb = xgb.train(params=self.params,
                             dtrain=dtrain,
                             num_boost_round=self.num_round,
                             early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=self.verbose)
        return

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = pd.DataFrame(data=self.xgb.predict(dtest), index=X.index)
        return preds


# Regressor Wrapper Class
class XGBRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    XGBRegressor in xgboost for sklearn doesnt have ALL parameters accessible, a simple wrapper to expose them
    params = {colsample_bytree=0.9,
              learning_rate=0.01,
              max_depth=5,
              min_child_weight=1,
              n_estimators=300,
              nthread=-1,
              objective='binary:logistic',
              seed=0,
              silent=True,
              subsample=0.8}
    a = XGBRegressor(params=params)
    a.fit(X_train, y_train)
    """

    def __init__(self, params={}, num_round=50, eval_metric=None, early_stopping_rounds=None, verbose_eval=True):
        self.params = params
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.verbose = verbose_eval
        self.xgb = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.xgb = xgb.train(params=self.params,
                             dtrain=dtrain,
                             num_boost_round=self.num_round,
                             early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=self.verbose)
        return

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        preds = pd.DataFrame(self.xgb.predict(dtest))
        return preds

