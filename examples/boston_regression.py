"""
Boston house prices example of usage for regression problems
A simple example of how to use gestalt
  1. Use custom wrapper for XGB
  2. Create a set of Base Classifiers
  3. Hyperparameter tune Base Classifiers (TODO)
  4. Run a stack ensemble

"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

########################################################################################################################
# Grab data and save base
data, target = load_boston(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)

########################################################################################################################

# Test out Gestalt.
import pandas as pd
from sklearn.model_selection import KFold
from gestalt.stackers.stacking import GeneralisedStacking
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss

skf = KFold(n_splits=3, random_state=42, shuffle=True)
b_classifiers = [RandomForestRegressor(n_estimators=100, n_jobs=18, random_state=42)]

for stype in ['t', 'cv']:
    b_cancer = GeneralisedStacking(base_estimators=b_classifiers, estimator_type='classification', feval=log_loss,
                                   stack_type=stype, folds_strategy= skf)
    b_cancer.fit(pd.DataFrame(train_x), pd.DataFrame(target_x))
