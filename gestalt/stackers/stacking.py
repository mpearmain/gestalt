#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from gestalt.utils.fold_splitter import split_folds


class GeneralisedStacking:
    """
    A general driver for a stacking framework, the simple idea is a common interface for running stackers in python
    based on the same set of folds created from an sklearn.model_selection type (say Kfold)
    Within Gestalt we support three data types for running stackers - Dense pandas DataFrames numpy array and
    scipy sparse csr_matrix
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
        # Check that the estimator is a dictionary of estimators and names.
        if not isinstance(base_estimators_dict, dict):
            raise ValueError("\nbase_estimators_dict must be a dictionary of estimator and name e.g\n"
                             "estimators = {RandomForestClassifier(n_estimators=123, random_state=42): 'RFC1',\n"
                             "              RandomForestClassifier(n_estimators=321, random_state=56), 'RFC2'}")
        # Check that the estimator type has been set.
        if not (estimator_type in ('classification', 'regression')):
            raise ValueError("estimator_type must be either 'classification', 'regression'")
        # Check that the stack type has been set.
        if not (stack_type in ('s', 't', 'st', 'cv')):
            raise ValueError("stack_type must be either 's', 't', 'st', or 'cv'")

        self.base_estimators = list(base_estimators_dict.keys())
        self.base_estimators_names = list(base_estimators_dict.values())
        self.folds_strategy = folds_strategy
        self.estimator_type = estimator_type
        self.stack_type = stack_type
        self.feval = feval
        self.stacking_train = None
        self.num_classes = None
        self.fold_estimators = {}

    def fit(self, X, y):
        """ A generic fit method for meta stacking.
        Under the hood we call one of four _fit_X functions for the specific use case.
        :param X: Train dataset
        :param y: target
        """
        # Generate an index to be used to build the storage data frame.
        # If the data came from a pd.DataFrame we use that index
        index_X = np.arange(X.shape[0])
        if isinstance(X, pd.DataFrame):
            index_X = X.index

        if isinstance(y, pd.DataFrame) and self.estimator_type is 'classification':
             y = y.values.ravel()

        if self.estimator_type is 'classification':
            self.num_classes = len(np.unique(y))

        # Create a holding dataframe to populate with out of fold predictions.
        if self.estimator_type is 'classification' and self.num_classes > 2:
            # Generate the multiclass stacking trainset - as many cols as models * classes.
            self.stacking_train = pd.DataFrame(np.nan,
                                               index=index_X,
                                               columns=[model_name + '_class_' + str(i)
                                                        for i in range(self.num_classes)
                                                        for model_name in self.base_estimators_names])
        # Create a single col per model for binary classification or regression problems.
        elif self.estimator_type is 'regression' or self.num_classes == 2:
            self.stacking_train = pd.DataFrame(np.nan, index=index_X, columns=self.base_estimators_names)

        print("Fitting type", self.stack_type, "stack.")
        for model_no in range(len(self.base_estimators)):
            print("Running Model (", self.base_estimators_names[model_no], ")",
                  model_no + 1, "of", len(self.base_estimators))
            if self.stack_type is 't':
                self._fit_t(X, y, model_no)
            elif self.stack_type is 'cv':
                self._fit_cv(X, y, model_no)
            elif self.stack_type is 'st':
                self._fit_st(X, y, model_no)
            elif self.stack_type is 's':
                self._fit_s(X, y, model_no)
        return

    def _fit_t(self, X, y, model_no):
        # We only run against the full dataset i.e no 'cv' information is available.
        self.base_estimators[model_no].fit(X, y)
        return

    def _fit_cv(self, X, y, model_no):
        # We only look at cv across folds - no storing of models or results with 'cv'
        evals = []
        i = 0
        for train_index, test_index in self.folds_strategy.split(X, y):
            # Loop over the different folds.
            X_train, X_test, y_train, y_test = split_folds(train_index, test_index, X, y)

            # Fit on each fold for each model.
            # We'll add a try condition here for using x_test as a validation set - useful of XGB early stopping.
            try:
                self.base_estimators[model_no].fit(X_train, y_train, X_test, y_test)
            except:
                pass
            else:
                self.base_estimators[model_no].fit(X_train, y_train)
            if self.estimator_type is 'regression':
                predicted_y = self.base_estimators[model_no].predict(X_test)
            elif self.estimator_type is 'classification':
                predicted_y = self.base_estimators[model_no].predict_proba(X_test)
                if self.num_classes is 2 and 'sklearn' in str(type(self.base_estimators[model_no])):
                    predicted_y = predicted_y[:, 1]

            if self.feval is not None:
                fold_score = self.feval(y_test, predicted_y)
                evals.append(fold_score)
                print('Fold{}: {}'.format(i + 1, evals[i]))
                i += 1
        print('CV Mean: ', np.mean(evals), ' Std: ', np.std(evals))
        return

    def _fit_st(self, X, y, model_no):
        # Fit a model that stacks for CV folds, predicts the out-of-fold rows for the X and then runs a full fit on
        # the data to use for preditions.
        evals = []
        i = 0

        for train_index, test_index in self.folds_strategy.split(X, y):
            # Loop over the different folds.
            X_train, X_test, y_train, y_test = split_folds(train_index, test_index, X, y)

            # Fit on each fold for each model.
            # We'll add a try condition here for using x_test as a validation set - useful of XGB early stopping.
            try:
                self.base_estimators[model_no].fit(X_train, y_train, X_test, y_test)
            except:
                self.base_estimators[model_no].fit(X_train, y_train)
            # Predict on the out of fold set
            if self.estimator_type is 'regression':
                predicted_y = self.base_estimators[model_no].predict(X_test)
                self.stacking_train.ix[test_index, self.base_estimators_names[model_no]] = predicted_y
            elif self.estimator_type is 'classification':
                predicted_y = self.base_estimators[model_no].predict_proba(X_test)
                if self.num_classes == 2:
                    if 'sklearn' in str(type(self.base_estimators[model_no])):
                        predicted_y = predicted_y[:, 1]
                    self.stacking_train.ix[test_index, self.base_estimators_names[model_no]] = predicted_y
                elif self.num_classes > 2:
                    self.stacking_train.ix[test_index, [self.base_estimators_names[model_no] +
                                                        '_class_' + str(i) for i in
                                                        range(self.num_classes)]] = predicted_y
            # Evaluate the Folds
            if self.feval is not None:
                fold_score = self.feval(y_test, predicted_y)
                evals.append(fold_score)
                print('Fold{}: {}'.format(i + 1, evals[i]))
                i += 1

        print('CV Mean: ', np.mean(evals), ' Std: ', np.std(evals))
        # Finally fit against all the data
        self._fit_t(X, y, model_no)
        return

    def _fit_s(self, X, y, model_no):
        # Fit a model that stacks for CV folds, predicts the out-of-fold rows for X, and then runs a predict on the
        # test set, the final test set prediction is the average from all fold models.
        evals = []
        fold_fits = {}
        i = 0

        for train_index, test_index in self.folds_strategy.split(X, y):
            # Loop over the different folds.
            X_train, X_test, y_train, y_test = split_folds(train_index, test_index, X, y)

            # Fit on each fold for each model.
            # We'll add a try condition here for using x_test as a validation set - useful of XGB early stopping.
            try:
                self.base_estimators[model_no].fit(X_train, y_train, X_test, y_test)
            except:
                self.base_estimators[model_no].fit(X_train, y_train)

            # Predict on the out of fold set
            if self.estimator_type is 'regression':
                predicted_y = self.base_estimators[model_no].predict(X_test)
                self.stacking_train.ix[test_index, self.base_estimators_names[model_no]] = predicted_y
            elif self.estimator_type is 'classification':
                predicted_y = self.base_estimators[model_no].predict_proba(X_test)
                if self.num_classes == 2:
                    if 'sklearn' in str(type(self.base_estimators[model_no])):
                        predicted_y = predicted_y[:, 1]
                    self.stacking_train.ix[test_index, self.base_estimators_names[model_no]] = predicted_y
                elif self.num_classes > 2:
                    self.stacking_train.ix[test_index,
                                           [self.base_estimators_names[model_no] +
                                            '_class_' + str(j) for j in range(self.num_classes)]] = predicted_y

            # Finally save the base_estimator.fit object for each fold of the data set.
            # In predict we need to loop through these to get an average prediction for the test set.
            # We create a model specific dictionary and we append each fold to this
            fold_fits[self.base_estimators_names[model_no] + 'fold' + str(i)] = self.base_estimators[model_no]
            # Evaluate the Folds
            if self.feval is not None:
                fold_score = self.feval(y_test, predicted_y)
                evals.append(fold_score)
                print('Fold{}: {}'.format(i + 1, evals[i]))
            i += 1

        print('CV Mean: ', np.mean(evals), ' Std: ', np.std(evals))
        # Last part add to the fold estimators
        self.fold_estimators[self.base_estimators_names[model_no]] = fold_fits
        return

    def predict(self, X):
        index_X = np.arange(X.shape[0])
        if isinstance(X, pd.DataFrame):
            index_X = X.index

        stacking_predict_data = pd.DataFrame(np.nan, index=index_X, columns=self.base_estimators_names)

        for model_no in range(len(self.base_estimators)):
            print("Predicting Model (", self.base_estimators_names[model_no], ")",
                  model_no + 1, "of", len(self.base_estimators))
            if self.stack_type is 't':
                self._predict_t(X, model_no, stacking_predict_data)
            elif self.stack_type is 'cv':
                print("No predictions available for CV type, try 't', 's', or 'st'")
            elif self.stack_type is 'st':
                # This uses the same function to train as predict t so we can reuse the same function.
                self._predict_t(X, model_no, stacking_predict_data)
            elif self.stack_type is 's':
                self._predict_s(X, model_no, stacking_predict_data)
        return stacking_predict_data

    def predict_proba(self, X):
        if self.estimator_type is not 'classification':
            raise ValueError("predit_proba can only be called on classification estimator_type problems")
        # Generate an index to be used to build the storage data frame.
        # If the data came from a pd.DataFrame we use that index
        index_X = np.arange(X.shape[0])
        if isinstance(X, pd.DataFrame):
            index_X = X.index

        # Create a holding dataframe to populate with out of fold predictions.
        if self.num_classes > 2:
            # Generate the multiclass stracking trainset - as many cols as models * classes.
            stacking_predict_proba_data = pd.DataFrame(np.nan,
                                                       index=index_X,
                                                       columns=[model_name + '_class_' + str(i)
                                                                for i in range(self.num_classes)
                                                                for model_name in self.base_estimators_names])
        # Create a single col per model for binary classification problems.
        elif self.num_classes == 2:
            stacking_predict_proba_data = pd.DataFrame(np.nan, index=index_X, columns=self.base_estimators_names)

        for model_no in range(len(self.base_estimators)):
            print("Predicting Model (", self.base_estimators_names[model_no], ")",
                  model_no + 1, "of", len(self.base_estimators))
            if self.stack_type is 't':
                self._predict_proba_t(X, model_no, stacking_predict_proba_data)
            elif self.stack_type is 'cv':
                print("No predictions available for CV type, try 't', 's', or 'st'")
                stacking_predict_proba_data = None
            elif self.stack_type is 'st':
                # This uses the same function to train as predict t so we can reuse the same function.
                self._predict_proba_t(X, model_no, stacking_predict_proba_data)
            elif self.stack_type is 's':
                self._predict_proba_s(X, model_no, stacking_predict_proba_data)

        return stacking_predict_proba_data

    def _predict_t(self, X, model_no, stacking_predict_data):
        # Predict on the test set of data X
        predicted_y = self.base_estimators[model_no].predict(X)
        stacking_predict_data.ix[:, self.base_estimators_names[model_no]] = predicted_y
        return stacking_predict_data

    def _predict_proba_t(self, X, model_no, stacking_predict_proba_data):
        predicted_y = self.base_estimators[model_no].predict_proba(X)
        if self.num_classes == 2:
            if 'sklearn' in str(type(self.base_estimators[model_no])):
                predicted_y = predicted_y[:, 1]
            stacking_predict_proba_data.ix[:, self.base_estimators_names[model_no]] = predicted_y
        elif self.num_classes > 2:
            multicol_names = \
                [self.base_estimators_names[model_no] + '_class_' + str(i) for i in range(self.num_classes)]
            stacking_predict_proba_data.ix[:, multicol_names] = predicted_y
        return stacking_predict_proba_data

    def _predict_s(self, X, model_no, stacking_predict_data):
        # To get the averaged predicted_y we have to loop through all the base_estimators for that model from each fold
        # and then take the mean average.
        predicted_y = 0
        for estimator in self.fold_estimators[self.base_estimators_names[model_no]].values():
            predicted_y += estimator.predict(X)
        predicted_y /= self.folds_strategy.n_splits

        stacking_predict_data.ix[:, self.base_estimators_names[model_no]] = predicted_y

        return stacking_predict_data

    def _predict_proba_s(self, X, model_no, stacking_predict_proba_data):
        # To get the averaged predicted_y we have to loop through all the base_estimators for that model from each fold
        # and then take the mean average.
        predicted_y = 0
        for estimator in self.fold_estimators[self.base_estimators_names[model_no]].values():
            predicted_y += estimator.predict_proba(X)
        predicted_y /= self.folds_strategy.n_splits

        if self.num_classes == 2:
            if 'sklearn' in str(type(self.base_estimators[model_no])):
                predicted_y = predicted_y[:, 1]
            stacking_predict_proba_data.ix[:, self.base_estimators_names[model_no]] = predicted_y
        elif self.num_classes > 2:
            multicol_names = \
                [self.base_estimators_names[model_no] + '_class_' + str(i) for i in range(self.num_classes)]
            stacking_predict_proba_data.ix[:, multicol_names] = predicted_y
        return stacking_predict_proba_data

    @property
    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict
        """
        return self.stacking_train
