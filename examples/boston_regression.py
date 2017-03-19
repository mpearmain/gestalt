"""
Boston house prices example of usage for regression problems
A simple example of how to use gestalt:
We provide three examples:
A. Pandas - X and y are both pandas DataFrames
  1. Use custom wrapper for XGB to use any paras we choose, use a random forest from scikit-learn
  2. Create a set of Base Classifiers
  3. Run all for different types of stacker.

B. Numpy - X and y are both numpy arrays
  1. Use custom wrapper for XGB to use any paras we choose, use a random forest from scikit-learn
  2. Create a set of Base Classifiers
  3. Run all for different types of stacker.

C. Sparse CSR - X is a sparse csr matrix and y a numpy arrays.
  1. Use custom wrapper for XGB to use any paras we choose, use a random forest from scikit-learn
  2. Create a set of Base Classifiers
  3. Run all for different types of stacker.

"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy import sparse
########################################################################################################################
# Grab data
print("\n Pandas Examples")
data, target = load_boston(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)

########################################################################################################################

# Test out Gestalt.
import pandas as pd
from sklearn.model_selection import KFold
from gestalt.stackers.stacking import GeneralisedStacking
from sklearn.ensemble import RandomForestRegressor
from gestalt.estimator_wrappers.wrap_xgb import XGBRegressor
from sklearn.metrics import mean_squared_error as mse

skf = KFold(n_splits=3, random_state=42, shuffle=True)
estimators = {RandomForestRegressor(n_estimators=100, n_jobs=8, random_state=42): 'RFR1',
              XGBRegressor(num_round=50, verbose_eval=False, params={'silent': 1}): 'XGB1'}

for stype in ['t', 'cv', 'st', 's']:
    boston = GeneralisedStacking(base_estimators_dict=estimators,
                                 estimator_type='regression',
                                 feval=mse,
                                 stack_type=stype,
                                 folds_strategy=skf)
    boston.fit(pd.DataFrame(train_x), pd.DataFrame(target_x))
    boston.predict(pd.DataFrame(test_x))

###
print("\n Numpy example")
for stype in ['t', 'cv', 'st', 's']:
    boston = GeneralisedStacking(base_estimators_dict=estimators,
                                 estimator_type='regression',
                                 feval=mse,
                                 stack_type=stype,
                                 folds_strategy=skf)
    boston.fit(train_x, target_x)
    boston.predict(test_x)

##
print("\nSparse CSR example")
X = sparse.csr_matrix(train_x)
test_x = sparse.csr_matrix(test_x)

for stype in ['t', 'cv', 'st', 's']:
    boston = GeneralisedStacking(base_estimators_dict=estimators,
                                 estimator_type='regression',
                                 feval=mse,
                                 stack_type=stype,
                                 folds_strategy=skf)
    boston.fit(X, target_x)
    boston.predict(test_x)