from itertools import combinations

import numpy as np
import pandas as pd


class StackingBayesEncoder:
    """
    To facilitate bayesian encoding across stacks
    It only provides fit_transform function.
    """

    def __init__(self, folds_strategy):
        """
        :param folds_strategy: A folds generator object from sklearn.model_selection. (sklearn v0.18)
        """
        self.xfolds = folds_strategy

    def fit_transform(self, X, y, encode_list, target_col, levels=1):
        """
        A generic fit method for meta stacking for bayesian encoding.

        :param X: A pandas dataset (train).
        :param y: A pandas dataset (test).
        :param encode_list: The list of single cols in the train and test sets to encode.
        :param target_col: The numeric col to encode against, float.
        :param levels: The number of levels to encode to, all single (1), pairwise (2), triplets (3)

        """
        # We first need calculate the number of encodings we are going to make based on the combination levels.
        encode_combinations = list(combinations(encode_list, levels))
        encode_train = pd.DataFrame(data=None, index=X.index)
        encode_test = pd.DataFrame(data=None, index=y.index)
        y_idx = y.index
        # For each fold in the data set
        for traincv, testcv in self.xfolds.split(X):
            # First create the different datasets to encode.
            encode_X = X.ix[traincv]
            encode_y = X.ix[testcv]

            # iterate through the combinations, first extract the cols from the tuple.
            for encode_row in encode_combinations:
                encode_cols = []
                [encode_cols.append(j) for j in encode_row]

                aggr_funcs = ["mean", "median"]

                meanDF = pd.DataFrame(encode_X.groupby(encode_cols)[target_col].aggregate(aggr_funcs))
                meanDF = meanDF.reset_index()

                global_mean = encode_X.groupby(encode_cols)[target_col].mean().mean()
                global_median = encode_X.groupby(encode_cols)[target_col].median().median()

                label = target_col
                for i in encode_cols:
                    label = label + i
                print("Getting", label, "wise demand..")

                dfcols = [[col for col in encode_cols], [i + label for i in aggr_funcs]]
                meanDF.columns = [item for sublist in dfcols for item in sublist]

                # We only care about the values in this stack.
                encode_y = pd.merge(encode_y, meanDF, on=encode_cols, how="left")
                # fill any missing values (in y not in X)
                encode_y['mean' + label].fillna(global_mean, inplace=True)
                encode_y['median' + label].fillna(global_median, inplace=True)

                encode_y.index = testcv

                #  Create col if not present and add y
                for col in dfcols[1]:
                    if col not in encode_train:
                        encode_train[col] = np.nan
                    encode_train.ix[encode_y.index, col] = encode_y[col]


        # Repeat for all the data and create the test set values
        # iterate through the combinations, first extract the cols from the tuple.
        for encode_row in encode_combinations:
            encode_cols = []
            [encode_cols.append(j) for j in encode_row]

            aggr_funcs = ["mean", "median"]

            meanDF = pd.DataFrame(X.groupby(encode_cols)[target_col].aggregate(aggr_funcs))
            meanDF = meanDF.reset_index()

            global_mean = X.groupby(encode_cols)[target_col].mean().mean()
            global_median = X.groupby(encode_cols)[target_col].median().median()

            label = target_col
            for i in encode_cols:
                label = label + i
            print("Getting", label, "wise demand..")

            dfcols = [[col for col in encode_cols], [i + label for i in aggr_funcs]]
            meanDF.columns = [item for sublist in dfcols for item in sublist]

            # We only care about the values in this stack.
            y = pd.merge(y, meanDF, on=encode_cols, how="left")
            # fill any missing values (in y not in X)
            y['mean' + label].fillna(global_mean, inplace=True)
            y['median' + label].fillna(global_median, inplace=True)

            y.index = y_idx

            #  Create col if not present and add y
            for col in dfcols[1]:
                if col not in encode_test:
                    encode_test[col] = np.nan
                encode_test.ix[encode_test.index, col] = y[col]

        return encode_train, encode_test