#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy import sparse

import gestalt.stackers.pandas_stacking as pd_stack
import gestalt.stackers.scipy_csr_stacking as csr_stack


class GeneralisedStacking:
    """
    A general driver for a stacking framework, the simple idea is a common interface for running stackers in python
    based on the same set of folds created from an sklearn.model_selection type (say Kfold)
    Within Gestalt we support two data types for running stackers - Dense pandas DataFrames and scipy sparse csr_matrix
    """

    def __init__(self, base_estimators_dict, folds_strategy, estimator_type, stack_type, feval):
        """
        :param base_estimators_dict: A dictionary of estimator type and name (this avoids the awkward arbitrary naming of
                                columns when storing meta level information).
        :param folds_strategy: An sklearn.model_selection generator for either Kfolds or Stratified Kfold.
        :param estimator_type: Classification or Regression (if classification we use, nuniques to determine multi-class
                            or binary prediction.
        :param stack_type: Defines the type of model stacking we want to run.
        's': Stack averaging. Saving an oof prediction for the train set and an average of test prediction based on
             per fold estimator fits.
             -- Useful for quasi-bagging of results and important if feature transforms have been performed at a
                fold level (e.g bayesian encoding in transformers) that the same folds are used in building the
                meta-models. -- DATA LEAKAGE COULD OCCUR!
        't': Training all data and predict test.
             -- Least powerful, it essentially runs fit and then predicts, no CV available but faster than 'st'
        'st': Stacking and then training on all data to predict test using save final model with cross-validation
              -- Use this for train / test splits of data setup like time series
        'cv': Only cross validation without saving the prediction.
              -- Use this for the final level ensembler to get a feel for the loss.
        :param feval: The evaluation function e.g sklearn.metrics.log_loss.
                      This function is expected to have the following inputs feval(y_prob, y_true)
        """
        self.stacking = None
        # Check that the estimator is a dictionary of estimators and names.
        if not isinstance(base_estimators_dict, dict):
            raise ValueError("\nbase_estimators_dict must be a dictionary of estimator and name e.g\n"
                             "estimators = {RandomForestClassifier(n_estimators=123, random_state=42): 'RFC1',\n"
                             "              RandomForestClassifier(n_estimators=321, random_state=56), 'RFC2'}")
        self.base_estimators_dict = base_estimators_dict
        self.folds_strategy = folds_strategy
        self.feval = feval

        # Check that the estimator type has been set.
        if not (estimator_type in ('classification', 'regression')):
            raise ValueError("estimator_type must be either 'classification', 'regression'")
        self.estimator_type = estimator_type

        # Check that the stack type has been set.
        if not (stack_type in ('s', 't', 'st', 'cv')):
            raise ValueError("stack_type must be either 's', 't', 'st', or 'cv'")
        self.stack_type = stack_type

    def fit(self, X, y):
        """

        :param X: The source training data, either pandas or scipy csr
        :param y: The source target variable
        :return:
        """

        if isinstance(X, pd.DataFrame):
            # y is a dataframe object with one col.
            if isinstance(y, pd.DataFrame) & y.shape[1] is 1:
                self.stacking = pd_stack.GeneralisedStacking(self.base_estimators_dict, self.folds_strategy,
                                                             self.estimator_type, self.stack_type, self.feval)
        if isinstance(X, sparse.csr_matrix):
            if isinstance(y, np.array):
                self.stacking = csr_stack.GeneralisedStacking(self.base_estimators_dict, self.folds_strategy,
                                                              self.estimator_type, self.stack_type, self.feval)
        self.stacking.fit(X, y)

    def predict(self, X):
        if self.estimator_type is not 'regression':
            raise ValueError("Predit can only be called on regression estimator_type problems")
        if self.stacking is None:
            raise ValueError("Fit must have been called before you can use `predict`")
        return self.stacking.predict(X)

    def predict_proba(self, X):
        if self.estimator_type is not 'classification':
            raise ValueError("predit_proba can only be called on classification estimator_type problems")
        if self.stacking is None:
            raise ValueError("Fit must have been called before you can use `predict_proba`")
        return self.stacking.predict_proba(X)

    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict
        """
        return self.stacking.meta_train()
