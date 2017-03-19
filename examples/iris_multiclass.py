"""
Iris example of multiclass usage
A simple example of how to use gestalt:
We provide three examples:
A. Pandas - X and y are both pandas DataFrames
  1. Use custom wrapper for XGB to use any paras we choose, use custom R wrapper for ranger used with pandas,
     use a random forest from scikit-learn
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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from gestalt.utils.multiclass_logloss import mlogloss
import pandas as pd
from scipy import sparse
########################################################################################################################
# Grab data
data, target = load_iris(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)
X = pd.DataFrame(train_x)
y = pd.DataFrame(target_x, columns=['target'])
test_x = pd.DataFrame(test_x)
test_y = pd.DataFrame(target_y, columns=['target'])
########################################################################################################################

# Test out Gestalt.

from sklearn.model_selection import KFold
from gestalt.stackers.stacking import GeneralisedStacking
from gestalt.estimator_wrappers.wrap_xgb import XGBClassifier
from gestalt.estimator_wrappers.wrap_r_ranger import RangerClassifier
from sklearn.ensemble import RandomForestClassifier

skf = KFold(n_splits=3, random_state=42, shuffle=True)
estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFC1',
              XGBClassifier(num_round=50,
                            verbose_eval=False,
                            params={'objective': 'multi:softprob',
                                    'num_class': 3,
                                    'silent': 1}):
                  'XGB1',
              RangerClassifier(num_trees=50, num_threads=8, seed=42): 'Ranger1'}
print("\nPandas Test")
for stype in ['t', 'cv', 'st', 's']:
    iris = GeneralisedStacking(base_estimators_dict=estimators,
                               estimator_type='classification',
                               feval=mlogloss,
                               stack_type=stype,
                               folds_strategy=skf)
    iris.fit(X, y)
    iris.predict_proba(test_x)

################################################################
# numpy test
print("\nNumpy Test")

########################################################################################################################
# Grab data
data, target = load_iris(return_X_y=True)
X, test_x, y, test_y = train_test_split(data, target, test_size=0.1, random_state=42)
########################################################################################################################

estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFC1',
              XGBClassifier(num_round=50,
                            verbose_eval=False,
                            params={'objective': 'multi:softprob',
                                    'num_class': 3,
                                    'silent': 1}): 'XGB1',}

for stype in ['t', 'cv', 'st', 's']:
    iris = GeneralisedStacking(base_estimators_dict=estimators,
                               estimator_type='classification',
                               feval=mlogloss,
                               stack_type=stype,
                               folds_strategy=skf)
    iris.fit(X, y)
    iris.predict_proba(test_x)

################################################################
# Sparse test
print("\nSparse Test")

########################################################################################################################
# Grab data
data, target = load_iris(return_X_y=True)
X, test_x, y, test_y = train_test_split(data, target, test_size=0.1, random_state=42)
X = sparse.csr_matrix(X)
test_x = sparse.csr_matrix(test_x)
# y values remain as numpy arrays

########################################################################################################################

estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFC1',
              XGBClassifier(num_round=50,
                            verbose_eval=False,
                            params={'objective': 'multi:softprob',
                                    'num_class': 3,
                                    'silent': 1}): 'XGB1',}

for stype in ['t', 'cv', 'st', 's']:
    iris = GeneralisedStacking(base_estimators_dict=estimators,
                               estimator_type='classification',
                               feval=mlogloss,
                               stack_type=stype,
                               folds_strategy=skf)
    iris.fit(X, y)
    iris.predict_proba(test_x)



