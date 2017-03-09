#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CV output in Base needs to be fixed for blended test output
"""
import numpy as np
import pandas as pd
# BaseEstimator
from sklearn.base import BaseEstimator


# BaseModel Class
class BaseModel(BaseEstimator):
    """
    Parameters of fit
    ----------

    problem_type = 'classification'
    classification_type = 'multi-class'
    eval_type = log_loss
    BaseModel.set_prob_type(problem_type, classification_type, eval_type)


    Example
    -------
    from models.base import BaseModel,
    from loaders.loader import load_pd
    from wrappers.wrap_xgb  import XGBClassifier
    FEATURE_LIST = {'train':('train.csv'),
                    'target': 'target.csv'
                    'test':('test.csv'),}
    PARAMS = {'n_estimator':700,'sub_sample': 0.8,'seed': 71}

    class ModelV1(BaseModel):
         def build_model(self):
         return XGBClassifier(**self.params)

    if __name__ == "__main__":
        skf = Kfold(folds=10, shuffle=True, random_state=42)
        m = ModelV1(name="level0_xgb",
                    flist=FEATURE_LIST,
                    params=PARAMS,
                    loader=load_pd
                    kind='st')
        m.run(fold_strategy=skf)
    """
    # Need to be set by BaseModel.set_prob_type()
    problem_type = None
    classification_type = None
    eval_type = None

    def __init__(self, name='', flist={}, params={}, loader=None, type=None):
        """
        :param name: Model name
        :param flist: Feature list
        :param params: Parameters
        :param loader: The loader type to use
        :param type: Kind of run()
        's': Stacking only. Saving an oof prediction({}_all_fold.csv) and average of test prediction based on
             fold-train models({}_test.csv).
             -- Useful for quasi-bagging of results and important if feature transforms have been performed at a
                fold level (e.g bayesian encoding in transformers)
        't': Training all data and predict test({}_test_FullTrainingData.csv).
             -- Least powerful, it essentially runs fit and then predicts, no CV available but faster than 'st'
        'st': Stacking and then training all data and predict test using save final model with cross-validation
              -- Use this for train / test splits of data (probably time series)
        'cv': Only cross validation without saving the prediction
              -- Use this for the final level ensembler to get a feel for the loss.
        """

        if BaseModel.problem_type == 'classification':
            if not (BaseModel.classification_type in ('binary', 'multi-class')):
                raise ValueError('Problem, Classification, and Evaluation types should be set before model defined')
            if BaseModel.eval_type is None:
                raise ValueError('Problem, and Evaluation types should be set before model defined')
        elif BaseModel.problem_type == 'regression':
            if BaseModel.eval_type is None:
                raise ValueError('Problem, and Evaluation types should be set before model defined')
        else:
            raise ValueError('Problem, Classification, and Evaluation types should be set before model defined')

        self.name = name
        self.flist = flist
        self.params = params
        self.loader = loader
        self.kind = type
        assert (self.kind in ['s', 't', 'st', 'cv'])

    @classmethod
    def set_prob_type(cls, problem_type, classification_type, eval_type):
        """

        :param problem_type: 'classification' or 'regression'
        :param classification_type: 'binary' or 'multi-class'
        :param eval_type:
        :return:
        """

        cls.problem_type = problem_type
        cls.classification_type = classification_type
        cls.eval_type = eval_type
        return

    def build_model(self):
        return None

    @staticmethod
    def make_multi_cols(num_class, name):
        """
        :param num_class: The number of classes in the multi-class classifier.
        :param name: The var name to use
        :return:
            Named cols for multi-class predictions"""
        cols = ['class' + str(i) + '_' + name for i in range(num_class)]
        return cols

    @classmethod
    def eval_pred(cls, y_true, y_pred):
        """
        :param y_true : array-like or label indicator matrix Ground truth (correct) labels for n_samples samples.
        :param y_pred : array-like of float, shape = (n_samples, n_classes) or (n_samples,)
                        Predicted probabilities, as returned by a classifier's predict_proba method.
        :param eval_type: The evaluation function to apply to the arrays.
        :return: Scalar : A loss value driven by the eval_type.
        """
        loss = cls.eval_type(y_true, y_pred)
        print("Loss Value: ", loss)
        return loss

    def run(self, fold_strategy):
        """
        :param fold_strategy: A folds generator object from sklearn.model_selection. (sklearn v0.18)
        """
        print('running model: {}'.format(self.name))
        # Remember X and y are pandas DataFrames!
        X, y, test = self.loader()
        num_class = y.ix[:, 0].nunique()  # only for multi-class classification

        if BaseModel.classification_type == 'multi-class':
            multi_cols = self.make_multi_cols(num_class, '{}_pred'.format(self.name))

        if self.kind == 't':
            clf = self.build_model()
            if 'sklearn' in str(type(clf)):
                y = y.ix[:, 0]
            clf.fit(X, y)
            if BaseModel.problem_type == 'classification':
                y_submission = clf.predict_proba(test)

                if BaseModel.classification_type == 'binary':
                    y_submission.to_csv(TEMP_PATH + '{}_test_FullTrainingData.csv'.format(self.name))

                elif BaseModel.classification_type == 'multi-class':
                    y_submission.columns = multi_cols
                    y_submission.to_csv(TEMP_PATH + '{}_test_FullTrainingData.csv'.format(self.name))

            elif BaseModel.problem_type == 'regression':
                y_submission = clf.predict(test)
                y_submission.to_csv(TEMP_PATH + '{}_test_FullTrainingData.csv'.format(self.name))
            return

        clf = self.build_model()
        print("Creating train and test sets for stacking.")

        # for binary
        if BaseModel.problem_type == 'regression' or BaseModel.classification_type == 'binary':
            dataset_blend_train = pd.DataFrame(np.nan, index=X.index, columns=self.name)
            dataset_blend_test = pd.DataFrame(np.nan, index=test.index, columns=self.name)

        # for multi-class
        elif BaseModel.classification_type == 'multi-class':
            dataset_blend_train = pd.DataFrame(np.nan, index=X.index, columns=multi_cols)
            dataset_blend_test = pd.DataFrame(np.nan, index=test.index, columns=multi_cols)

        # Start stacking
        evals = []
        for traincv, testcv in fold_strategy.split(X):
            # First create the different datasets to encode.
            X_train = X.iloc[traincv]
            X_test = X.iloc[testcv]
            y_train = y.iloc[traincv]
            y_test = y.iloc[testcv]

            # print X_train,y_train,X_test,y_test
            if 'sklearn' in str(type(clf)):
                y_train = y_train.ix[:, 0]
                clf.fit(X_train, y_train)
            else:
                clf.fit(X_train, y_train, X_test, y_test)

            if BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary':
                # if using the mean of the prediction of each fold print str(type(clf))
                if 'sklearn' in str(type(clf)):
                    ypred = clf.predict_proba(X_test)[:, 1]
                else:
                    ypred = clf.predict_proba(X_test)

            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
                if 'sklearn' in str(type(clf)):
                    ypred = pd.DataFrame(clf.predict_proba(X_test), columns=multi_cols)
                else:
                    ypred = clf.predict_proba(X_test)

            elif BaseModel.problem_type == 'regression':
                ypred = clf.predict(X_test)

            try:
                ypred.index = y_test.index
                dataset_blend_train.iloc[testcv] = ypred
            except:
                break

            evals.append(eval_pred(y_test, ypred, BaseModel.eval_type))

            # binary classification
            if BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'binary':
                # if using the mean of the prediction of each n_fold
                if 'sklearn' in str(type(clf)):
                    dataset_blend_test += clf.predict_proba(test)[:, 1]
                else:
                    dataset_blend_test += clf.predict_proba(test)

            # multi-class classification
            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
                # if using the mean of the prediction of each n_fold
                dataset_blend_test += clf.predict_proba(test)
                pass

            # regression
            elif BaseModel.problem_type == 'regression':
                # if using the mean of the prediction of each n_fold
                dataset_blend_test += clf.predict(test)

        dataset_blend_test /= fold_strategy.n_splits

        for i in range(fold_strategy.n_splits):
            print('Fold{}: {}'.format(i + 1, evals[i]))
        print('{} CV Mean: '.format(BaseModel.eval_type), np.mean(evals), ' Std: ', np.std(evals))

        # Saving
        if self.kind != 'cv':
            print('Saving results')
            if (BaseModel.problem_type == 'classification' and
                        BaseModel.classification_type == 'binary') or (BaseModel.problem_type == 'regression'):
                dataset_blend_train.to_csv(TEMP_PATH + '{}_all_fold.csv'.format(self.name))
                dataset_blend_test.to_csv(TEMP_PATH + '{}_test.csv'.format(self.name))

            elif BaseModel.problem_type == 'classification' and BaseModel.classification_type == 'multi-class':
                dataset_blend_train.to_csv(TEMP_PATH + '{}_all_fold.csv'.format(self.name))
                dataset_blend_test.to_csv(TEMP_PATH + '{}_test.csv'.format(self.name))

        if self.kind == 'st':
            # Stacking(cross-validation)
            clf = self.build_model()
            if 'sklearn' in str(type(clf)):
                y = y.ix[:, 0]
            clf.fit(X, y)
            if BaseModel.problem_type == 'classification':
                if BaseModel.classification_type == 'binary':
                    if 'sklearn' in str(type(clf)):
                        y_submission = clf.predict_proba(test)[:, 1]
                    else:
                        y_submission = clf.predict_proba(test)
                    y_submission.to_csv(TEMP_PATH + '{}_test_FullTrainingData.csv'.format(self.name))

                elif BaseModel.classification_type == 'multi-class':
                    if 'sklearn' in str(type(clf)):
                        y_submission = pd.DataFrame(clf.predict_proba(test), index=test.index, columns=multi_cols)
                    else:
                        y_submission = clf.predict_proba(test)
                        y_submission = test.index
                        y_submission.columns = multi_cols

                    y_submission.to_csv(TEMP_PATH + '{}_test_FullTrainingData.csv'.format(self.name))

            elif BaseModel.problem_type == 'regression':
                y_submission = clf.predict(test)
                y_submission.to_csv(TEMP_PATH + '{}_test_FullTrainingData.csv'.format(self.name))

        return
