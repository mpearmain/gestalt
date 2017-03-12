'''
Breast Cancer example of usage
A simple example of how to use gestalt for reading and running.\n
Below we import relavent modules to make the base dataset and save in ~/tmp/bcancer/.
We will nuke the dir at the end of the example but we need somwhere to put files
'''

import os

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Make the dir for storage of example. -- This assumes ~/tmp exists for the user!
PATH = os.path.expanduser("~") + "/tmp/bcancer/"
try:
    os.mkdir(PATH)
except:
    pass

########################################################################################################################
# Grab data and save base
data, target = load_breast_cancer(return_X_y=True)
train_x, test_x, target_x, target_y = train_test_split(data, target, test_size=0.1, random_state=42)
np.savetxt(PATH + "train_x.csv", train_x, delimiter=",")
np.savetxt(PATH + "test_x.csv", test_x, delimiter=",")
np.savetxt(PATH + "target_x.csv", target_x, delimiter=",")
np.savetxt(PATH + "target_y.csv", target_y, delimiter=",")
########################################################################################################################

# Test out Gestalt.

from gestalt.loaders import loader
import pandas as pd
from sklearn.model_selection import KFold
from gestalt.models.gestalt import Gestalt
from gestalt.wrappers.wrap_xgb import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import log_loss

skf = KFold(n_splits=10, random_state=42, shuffle=True)

# ----- Set problem type!! -----
problem_type = 'classification'
classification_type = 'binary'
eval_type = log_loss

Gestalt.set_prob_type(problem_type, classification_type, eval_type)

# Data
data_paths = {
    'train': (PATH + 'train_X.csv',),
    'target': (PATH + 'target_x.csv',),
    'test': (PATH + 'test_X.csv',),
}

# Models in level 1
PARAMS_XGB = {
    'colsample_bytree': 0.80,
    'learning_rate': 0.05,
    "eval_metric": "logloss",
    'max_depth': 6,
    'min_child_weight': 1,
    'nthread': -1,
    'seed': 42,
    'silent': 1,
    'subsample': 0.80}


class ModelXGB(Gestalt):
    def build_model(self):
        return XGBClassifier(params=self.params, num_round=100, early_stopping_rounds=10)


m = ModelXGB(loader=loader.load_pandas,
             type='cv',
             name="v1_level1",
             flist=data_paths,
             params=PARAMS_XGB)
m.run(fold_strategy=skf)



























