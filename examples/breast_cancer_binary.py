"""
Breast Cancer example of usage
A simple example of how to use gestalt:
We provide three examples:
A. PANDAS - X and y are both pandas DataFrames
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
import pandas as pd
from scipy import sparse
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

########################################################################################################################
# Grab data and save base
data, target = load_breast_cancer(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)
X = pd.DataFrame(train_x)
y = pd.DataFrame(target_x, columns=['target'])
test_x = pd.DataFrame(test_x)
test_y = pd.DataFrame(target_y, columns=['target'])
########################################################################################################################

# Test out Gestalt.

from sklearn.model_selection import KFold
from gestalt.stackers.stacking import GeneralisedStacking
from sklearn.ensemble import RandomForestClassifier
from gestalt.estimator_wrappers.wrap_xgb import XGBClassifier
from gestalt.estimator_wrappers.wrap_r_ranger import RangerClassifier
from sklearn.metrics import log_loss

skf = KFold(n_splits=3, random_state=42, shuffle=True)
# Base estimators come in the form of a dictionary of {estimator1:'name1', estimator2:'name2'}
# This makes life easy when naming the meta-learner dataset.
estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFR1',
              RangerClassifier(num_trees=50, num_threads=8, seed=42): 'Ranger1'}

print("\nPandas Test")
for stype in ['t', 'cv', 'st', 's']:
    print('\n')
    b_cancer = GeneralisedStacking(base_estimators_dict=estimators,
                                   estimator_type='classification',
                                   feval=log_loss,
                                   stack_type=stype,
                                   folds_strategy=skf)
    b_cancer.fit(X, y)
    if stype != 'cv':
        b_cancer.predict_proba(test_x)

################################################################
# numpy test
print("\nNumpy Test")
# Grab data
data, target = load_breast_cancer(return_X_y=True)
X, test_x, y, test_y = train_test_split(data, target, test_size=0.1, random_state=42)

# Remove ranger estimator as not build enhanced for numpy yet.
estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFR1',
              XGBClassifier(num_round=50,
                            verbose_eval=False,
                            params={'objective': 'binary:logistic',
                                    'silent': 1}): 'XGB1'}

for stype in ['t', 'cv', 'st', 's']:
    print('\n')
    b_cancer = GeneralisedStacking(base_estimators_dict=estimators,
                                   estimator_type='classification',
                                   feval=log_loss,
                                   stack_type=stype,
                                   folds_strategy=skf)
    b_cancer.fit(X, y)
    # Make predictions.
    if stype != 'cv':
        b_cancer.predict_proba(test_x)

###############################################################
#  scipy sparse test
print("\nScipy Sparse Test")
# Grab data
data, target = load_breast_cancer(return_X_y=True)
X, test_x, y, test_y = train_test_split(data, target, test_size=0.1, random_state=42)
X = sparse.csr_matrix(X)
test_x = sparse.csr_matrix(test_x)
# y values remain as numpy arrays

# Remove ranger estimator as not build enhanced for numpy yet.
estimators = {RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=42): 'RFR1',
              XGBClassifier(num_round=50,
                            verbose_eval=False,
                            params={'objective': 'binary:logistic',
                                    'silent': 1}):
              'XGB1'}

for stype in ['t', 'cv', 'st', 's']:
    print('\n')
    b_cancer = GeneralisedStacking(base_estimators_dict=estimators,
                                   estimator_type='classification',
                                   feval=log_loss,
                                   stack_type=stype,
                                   folds_strategy=skf)
    b_cancer.fit(X, y)
    # Make predictions.
    if stype != 'cv':
        b_cancer.predict_proba(test_x)