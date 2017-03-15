"""
Breast Cancer example of usage
A simple example of how to use gestalt
  1. Use custom wrapper for XGB
  2. Create a set of Base Classifiers
  3. Hyperparameter tune Base Classifiers
  4. Run a stack ensemble

"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

########################################################################################################################
# Grab data and save base
data, target = load_breast_cancer(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)

########################################################################################################################

# Test out Gestalt.
import pandas as pd
from sklearn.model_selection import KFold
from gestalt.stackers.stacking import Generalised_Stacking
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

skf = KFold(n_splits=3, random_state=42, shuffle=True)
b_classifiers = [RandomForestClassifier(n_estimators=100, n_jobs=18, random_state=42)]

b_cancer = Generalised_Stacking(base_estimators=b_classifiers, estimator_type='classification', feval=log_loss,
                                stack_type='cv', folds_strategy= skf)

b_cancer.fit(pd.DataFrame(train_x), pd.DataFrame(target_x))
