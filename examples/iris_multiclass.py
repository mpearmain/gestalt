"""
Iris example of multiclass usage
A simple example of how to use gestalt
  1. Use custom wrapper for XGB
  2. Create a set of Base Classifiers
  3. Hyperparameter tune Base Classifiers (TODO)
  4. Run a stack ensemble

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from gestalt.utils.multiclass_logloss import mlogloss

########################################################################################################################
# Grab data and save base
data, target = load_iris(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)

########################################################################################################################

# Test out Gestalt.
import pandas as pd
from sklearn.model_selection import KFold
from gestalt.stackers.stacking import GeneralisedStacking
from gestalt.estimator_wrappers.wrap_xgb import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

skf = KFold(n_splits=3, random_state=42, shuffle=True)
estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFC1',
              XGBClassifier(num_round=50,
                            verbose_eval=False,
                            params={'objective': 'multi:softprob',
                                    'num_class': 3,
                                    'silent': 1}):
                  'XGB1'}

for stype in ['t', 'cv', 'st', 's']:
    iris = GeneralisedStacking(base_estimators_dict=estimators,
                               estimator_type='classification',
                               feval=mlogloss,
                               stack_type=stype,
                               folds_strategy=skf)
    iris.fit(pd.DataFrame(train_x), pd.DataFrame(target_x))
    iris.predict_proba(pd.DataFrame(test_x))