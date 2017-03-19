import numpy as np
import pandas as pd
from gestalt.utils.fold_splitter import split_folds

class GeneralisedStacking:
    """
    A generalised stacking class specifically designed for use with pandas DataFrames and numpy arrays.
    """

    def __init__(self, base_estimators_dict, folds_strategy, estimator_type, stack_type, feval):
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
        if isinstance(y, pd.DataFrame):
            index_X = X.index
            if self.estimator_type is 'classification':
                self.num_classes = y.ix[:, 0].nunique()
        elif isinstance(y, np.ndarray):
            if self.estimator_type is 'classification':
                self.num_classes = len(np.unique(y))

        # Create a holding dataframe to populate with out of fold predictions.
        if self.estimator_type is 'classification' and self.num_classes > 2:
            # Generate the multiclass stracking trainset - as many cols as models * classes.
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
        if isinstance(y, pd.DataFrame):
            y = y.ix[:, 0].values
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
                                                    '_class_' + str(i) for i in range(self.num_classes)]] = predicted_y
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
        """
        :param X: The data to apply the fitted model from fit
        :return: The predicted value of the regression model
        """
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
        """
        :param X: The data to apply the fitted model from fit
        :return: The predicted value of the regression model
        """
        # Generate an index to be used to build the storage data frame.
        # If the data came from a pd.DataFrame we use that index
        index_X = np.arange(X.shape[0])
        if isinstance(X, pd.DataFrame):
            index_X = X.index

        # Create a holding dataframe to populate with out of fold predictions.
        if self.num_classes > 2:
            # Generate the multiclass stracking trainset - as many cols as models * classes.
            stacking_predict_data = pd.DataFrame(np.nan,
                                                 index=index_X,
                                                 columns=[model_name + '_class_' + str(i)
                                                          for i in range(self.num_classes)
                                                          for model_name in self.base_estimators_names])
        # Create a single col per model for binary classification problems.
        elif self.num_classes == 2:
            stacking_predict_data = pd.DataFrame(np.nan, index=index_X, columns=self.base_estimators_names)

        for model_no in range(len(self.base_estimators)):
            print("Predicting Model (", self.base_estimators_names[model_no], ")",
                  model_no + 1, "of", len(self.base_estimators))
            if self.stack_type is 't':
                self._predict_proba_t(X, model_no, stacking_predict_data)
            elif self.stack_type is 'cv':
                print("No predictions available for CV type, try 't', 's', or 'st'")
                stacking_predict_data = None
            elif self.stack_type is 'st':
                # This uses the same function to train as predict t so we can reuse the same function.
                self._predict_proba_t(X, model_no, stacking_predict_data)
            elif self.stack_type is 's':
                self._predict_proba_s(X, model_no, stacking_predict_data)

        return stacking_predict_data

    def _predict_t(self, X, model_no, stacking_predict_data):
        # Predict on the test set of data X
        predicted_y = self.base_estimators[model_no].predict(X)
        stacking_predict_data.ix[:, self.base_estimators_names[model_no]] = predicted_y
        return stacking_predict_data

    def _predict_proba_t(self, X, model_no, stacking_predict_data):
        predicted_y = self.base_estimators[model_no].predict_proba(X)
        if self.num_classes == 2:
            if 'sklearn' in str(type(self.base_estimators[model_no])):
                predicted_y = predicted_y[:, 1]
            stacking_predict_data.ix[:, self.base_estimators_names[model_no]] = predicted_y
        elif self.num_classes > 2:
            multicol_names = \
                [self.base_estimators_names[model_no] + '_class_' + str(i) for i in range(self.num_classes)]
            stacking_predict_data.ix[:, multicol_names] = predicted_y
        return stacking_predict_data

    def _predict_s(self, X, model_no, stacking_predict_data):
        # To get the averaged predicted_y we have to loop through all the base_estimators for that model from each fold
        # and then take the mean average.
        predicted_y = 0
        for estimator in self.fold_estimators[self.base_estimators_names[model_no]].values():
            predicted_y += estimator.predict(X)
        predicted_y /= self.folds_strategy.n_splits

        stacking_predict_data.ix[:, self.base_estimators_names[model_no]] = predicted_y

        return stacking_predict_data

    def _predict_proba_s(self, X, model_no, stacking_predict_data):
        # To get the averaged predicted_y we have to loop through all the base_estimators for that model from each fold
        # and then take the mean average.
        predicted_y = 0
        for estimator in self.fold_estimators[self.base_estimators_names[model_no]].values():
            predicted_y += estimator.predict_proba(X)
        predicted_y /= self.folds_strategy.n_splits

        if self.num_classes == 2:
            if 'sklearn' in str(type(self.base_estimators[model_no])):
                predicted_y = predicted_y[:, 1]
            stacking_predict_data.ix[:, self.base_estimators_names[model_no]] = predicted_y
        elif self.num_classes > 2:
            multicol_names = \
                [self.base_estimators_names[model_no] + '_class_' + str(i) for i in range(self.num_classes)]
            stacking_predict_data.ix[:, multicol_names] = predicted_y
        return stacking_predict_data

    @property
    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict
        """
        return self.stacking_train.copy()
