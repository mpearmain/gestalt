import numpy as np
import pandas as pd


class GeneralisedStacking:
    """
    A generalised stacking class specifically designed for use with dense pandas DataFrames.
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
        self.base_fit = []

    def fit(self, X, y):
        """ A generic fit method for meta stacking.
        Under the hood we call one of four _fit_X functions for the specific use case.
        :param X: Train dataset
        :param y: target
        """
        if self.estimator_type is 'classification':
            self.num_classes = y[0].nunique()

        # Create a holding dataframe to populate with out of fold predictions.
        if self.estimator_type is 'classification' and self.num_classes > 2:
            # Generate the multiclass stracking trainset - as many cols as models * classes.
            self.stacking_train = pd.DataFrame(np.nan,
                                               index=X.index,
                                               columns=[model_name + '_class_' + str(i)
                                                        for i in range(self.num_classes)
                                                        for model_name in self.base_estimators_names])
        # Create a single col per model for binary classification or regression problems.
        elif self.estimator_type is 'regression' or self.num_classes == 2:
            self.stacking_train = pd.DataFrame(np.nan, index=X.index, columns=self.base_estimators_names)

        for model_no in range(len(self.base_estimators)):
            print("Running Model (", self.base_estimators_names[model_no], ")",
                  model_no + 1, "of", len(self.base_estimators))
            print("Fitting type", self.stack_type, "stack.")
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
        y = y[0].values
        self.base_estimators[model_no].fit(X, y)
        return

    def _fit_cv(self, X, y, model_no):
        # We only look at cv across folds - no storing of models or results with 'cv'
        evals = []
        i = 0
        for traincv, testcv in self.folds_strategy.split(X, y):
            # Loop over the different folds.
            # First create the different datasets to encode.
            X_train = X.iloc[traincv]
            X_test = X.iloc[testcv]
            y_train = y.iloc[traincv][0].values
            y_test = y.iloc[testcv][0].values

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

        for traincv, testcv in self.folds_strategy.split(X, y):
            # Loop over the different folds.
            X_train = X.iloc[traincv]
            X_test = X.iloc[testcv]
            y_train = y.iloc[traincv][0].values
            y_test = y.iloc[testcv][0].values

            # Fit on each specific model.
            self.base_estimators[model_no].fit(X_train, y_train)
            # Predict on the out of fold set
            if self.estimator_type is 'regression':
                predicted_y = self.base_estimators[model_no].predict(X_test)
                self.stacking_train.ix[testcv, self.base_estimators_names[model_no]] = predicted_y

            elif self.estimator_type is 'classification':
                predicted_y = self.base_estimators[model_no].predict_proba(X_test)
                if self.num_classes == 2:
                    if 'sklearn' in str(type(self.base_estimators[model_no])):
                        predicted_y = predicted_y[:, 1]
                    self.stacking_train.ix[testcv, self.base_estimators_names[model_no]] = predicted_y
                elif self.num_classes > 2:
                    self.stacking_train.ix[testcv, [self.base_estimators_names[model_no] +
                                                    '_class_' + str(i) for i in range(self.num_classes)]] = predicted_y

            if self.feval is not None:
                fold_score = self.feval(y_test, predicted_y)
                evals.append(fold_score)
                print('Fold{}: {}'.format(i + 1, evals[i]))

                i += 1
        print('CV Mean: ', np.mean(evals), ' Std: ', np.std(evals))
        # Finally fit against all the data
        self._fit_t(X, y, model_no)
        return

    def _fit_s(self, X, y):
        pass

    def predict(self, X):
        """
        :param X: The data to apply the fitted model from fit
        :return: The predicted value of the regression model
        """
        stacking_predict_data = pd.DataFrame(np.nan, index=X.index, columns=self.colnames)

        for model_no in range(len(self.base_estimators)):
            stacking_predict_data.ix[:, model_no] = self.base_estimators[model_no].predict(X)
        return stacking_predict_data

    @property
    def meta_train(self):
        """
        A return method for the underlying meta data prediction from the training data set as a pandas dataframe
        Use the predict method to score new data for each classifier.
        :return: A pandas dataframe object of stacked predictions of predict
        """
        return self.stacking_train.copy()
