#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
When running stackers from sklearn the interface to the model api is standardized making it easy to simply add a
compiled model e.g.

PARAMS_RF = {'n_estimators': 500,
             'criterion': 'gini',
             'n_jobs': 8,
             'verbose': 0,
             'random_state': 407,
             'oob_score': True,
             }

class ModelRF(BaseModel):
    def build_model(self):
        return RandomForestClassifier(**self.params)

Other models outside of the sci-kit learn eco-system may have different api interfaces to call and run models
or (in the case of XGB and Keras) not all parameters are exposed in the sklearn interface they have made.

This is a set of common algorithms with wrappers to be able to run in our generic stacker class.

For each Wrapper class the output from predict or predict_proba should be a pandas dataset.

"""
import numpy as np
import pandas as pd
# Wrapper Class of Classifiers
from python.generalised_stacking.base import BaseModel
# BaseEstimator
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin

# Keras
from keras.utils import np_utils
from keras.callbacks import Callback

# XGBoost
import xgboost as xgb

from sklearn.metrics import log_loss

# Vowpal Wabbit
# Make sure libs are built  sudo apt-get install libboost-program-options-dev zlib1g-dev libboost-python-dev
# import vowpalwabbit as ww


class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}, eval_score=log_loss):
        if epoch % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = eval_score(self.y_val, y_pred)
            # logging.info("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))
            print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))


class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    (Example)
    from base import KerasClassifier
    class KerasModelV1(KerasClassifier):

        #Parameters for lerning
        #    batch_size=128,
        #    nb_epoch=100,
        #    verbose=1,
        #    callbacks=[],
        #    validation_split=0.,
        #    validation_data=None,
        #    shuffle=True,
        #    class_weight=None,
        #    sample_weight=None,
        #    normalize=True,
        #    categorize_y=False


        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=2, init='uniform'))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy',class_mode='binary')

            super(KerasModelV1, self).__init__(model,**params)

    KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                 validation_data=None, shuffle=True, class_weight=None, sample_weight=None, normalize=True,
                 categorize_y=True)
    KerasModelV1.fit(X_train, y_train,validation_data=[X_test,y_test])
    KerasModelV1.predict_proba(X_test)[:,1]
    """

    def __init__(self,
                 nn,
                 batch_size=128,
                 nb_epoch=100,
                 verbose=1,
                 callbacks=[],
                 validation_split=0.,
                 validation_data=None,
                 shuffle=True,
                 class_weight=None,
                 sample_weight=None,
                 normalize=True,
                 categorize_y=False):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        # set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, X_test=None, y_test=None):
        X = X.values  # Need for Keras
        y = y.values  # Need for Keras

        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1  # CAUTION!!!
            X = (X - self.mean) / self.std
        if self.categorize_y:
            y = np_utils.to_categorical(y)

        if X_test is not None:
            X_test = X_test.values  # Need for Keras
            y_test = y_test.values  # Need for Keras

            if self.normalize:
                X_test = (X_test - self.mean) / self.std
            if self.categorize_y:
                y_test = np_utils.to_categorical(y_test)

            self.validation_data = (X_test, y_test)
            self.callbacks = [IntervalEvaluation(validation_data=self.validation_data, interval=2)]
        else:
            self.validation_data = []

        # set initial weights
        self.nn.set_weights(self.init_weight)

        return self.nn.fit(X,
                           y,
                           batch_size=self.batch_size,
                           nb_epoch=self.nb_epoch,
                           validation_data=self.validation_data,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           validation_split=self.validation_split,
                           shuffle=self.shuffle,
                           class_weight=self.class_weight,
                           sample_weight=self.sample_weight)

    def predict_proba(self, X, batch_size=128, verbose=0):
        idx = X.index
        X = X.values  # Need for Keras
        if self.normalize:
            X = (X - self.mean) / self.std

        if BaseModel.classification_type == 'binary':
            preds = pd.DataFrame(self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose),
                                 index=idx)[:, 1]
        elif BaseModel.classification_type == 'multi-class':
            preds = pd.DataFrame(self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose),
                                 index=idx)
            return preds


class XGBClassifier(BaseEstimator, ClassifierMixin):
    '''
    XGBClassifier in xgboost for sklearn doesnt have ALL parameters accessible, a simple wrapper to expose them

    (Example)
    from base import XGBClassifier
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


class VWClassifier(BaseEstimator, ClassifierMixin):
    """
    Currently vowpalwabbit (the offical interface from VW github for python) doesnt support multiclass prediction.
     Short term lets skip this
    """

    def __init__(self):
        pass


# Regressor Wrapper Class


class KerasRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from base import KerasClassifier
    class KerasModelV1(KerasClassifier):

        #Parameters for lerning
        #    batch_size=128,
        #    nb_epoch=100,
        #    verbose=1,
        #    callbacks=[],
        #    validation_split=0.,
        #    validation_data=None,
        #    shuffle=True,
        #    class_weight=None,
        #    sample_weight=None,
        #    normalize=True,
        #    categorize_y=False


        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=1, init='uniform'))
            model.add(Activation('linear')
            # CAUSION
            # Change the output of last layer to 1
            # Change the loss to mse or mae
            # Using mse loss results in faster convergence

            model.compile(optimizer='rmsprop', loss='mean_absolute_error')#'mean_squared_error'

            super(KerasModelV1, self).__init__(model,**params)

    KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.,
                 validation_data=None, shuffle=True, class_weight=None, sample_weight=None, normalize=True,
                 categorize_y=True)
    KerasModelV1.fit(X_train, y_train,validation_data=[X_test,y_test])
    KerasModelV1.predict_proba(X_test)[:,1]
    """

    def __init__(self,
                 nn,
                 batch_size=128,
                 nb_epoch=100,
                 verbose=1,
                 callbacks=[],
                 validation_split=0.,
                 validation_data=None,
                 shuffle=True,
                 class_weight=None,
                 sample_weight=None,
                 normalize=True,
                 categorize_y=False,
                 random_sampling=None):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        self.random_sampling = random_sampling
        # set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, X_test=None, y_test=None):

        if self.random_sampling is not None:
            self.sampling_col = np.random.choice(range(X.shape[1]), self.random_sampling, replace=False)
            X = X.iloc[:, self.sampling_col].values  # Need for Keras
        else:
            X = X.values  # Need for Keras

        y = y.values  # Need for Keras

        if self.normalize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0) + 1
            X = (X - self.mean) / self.std

        if self.categorize_y:
            y = np_utils.to_categorical(y)

        if X_test is not None:
            X_test = X_test.values  # Need for Keras
            y_test = y_test.values  # Need for Keras
            if self.normalize:
                X_test = (X_test - self.mean) / self.std
            # if self.categorize_y:
            #    y_test = np_utils.to_categorical(y_test)

            self.validation_data = (X_test, y_test)

        else:
            self.validation_data = []

        # set initial weights
        self.nn.set_weights(self.init_weight)

        return self.nn.fit(X,
                           y,
                           batch_size=self.batch_size,
                           nb_epoch=self.nb_epoch,
                           validation_data=self.validation_data,
                           verbose=self.verbose,
                           callbacks=self.callbacks,
                           validation_split=self.validation_split,
                           shuffle=self.shuffle,
                           class_weight=self.class_weight,
                           sample_weight=self.sample_weight)

    def predict(self, X, batch_size=128, verbose=1):
        if self.random_sampling is not None:
            X = X.iloc[:, self.sampling_col].values
        else:
            X = X.values  # Need for Keras
        if self.normalize:
            X = (X - self.mean) / self.std

        return [pred_[0] for pred_ in self.nn.predict(X, batch_size=batch_size, verbose=verbose)]


class XGBRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from base import XGBClassifier
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


class VWRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
