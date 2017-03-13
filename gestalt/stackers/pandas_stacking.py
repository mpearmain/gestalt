import pandas as pd
import numpy as np

class Generalised_Stacking():
    """
    A generalised stacking class specifically designed for use with dense pandas DataFrames.
    """

    def __init__(self, base_estimators, folds_strategy, estimator_type, stack_type, feval):
        self.base_estimators = base_estimators
        self.folds_strategy = folds_strategy
        self.estimator_type = estimator_type
        self.stack_type = stack_type
        self.feval = feval

        # Build an empty pandas dataframe to store the meta results to.
        # As many rows as the folds data, as many cols as base regressors
        self.colnames = ["v" + str(n) for n in range(len(self.base_estimators))]
        self.stacking_train = None

    def fit(self, X, y, **kwargs):
        """ A generic fit method for meta stacking.
        :param X: Train dataset
        :param y: Train target
        :param kwargs: Any optional params to give the fit method, i.e in xgboost we may use eval_metirc='rmse'
        """
        self.stacking_train = pd.DataFrame(np.nan, index=X.index, columns=self.colnames)
        for model_no in range(len(self.base_estimators)):
            print("Running Model ", model_no + 1, "of", len(self.base_estimators))
            for traincv, testcv in self.folds_strategy:
                # Loop over the different folds.
                # First create the different datasets to encode.
                X_train = X.iloc[traincv]
                X_test = X.iloc[testcv]
                y_train = y.iloc[traincv]
                y_test = y.iloc[testcv]

                self.base_estimators[model_no].fit(X_train, y_train, **kwargs)
                predicted_y = self.base_estimators[model_no].predict(X_test)
                if self.feval is not None:
                    print("Current Score = ", self.feval(y_test, predicted_y))
                self.stacking_train.ix[testcv, model_no] = predicted_y
            # Finally fit against all the data
            self.base_estimators[model_no].fit(X, y, **kwargs)

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