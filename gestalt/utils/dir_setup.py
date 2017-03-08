# coding=utf-8
import os
# PATHS
# Change main folder name
DATA_PATH = ''
INPUT_PATH = 'input/'  # path of original data and fold_index
OUTPUT_PATH = 'output/'
TEMP_PATH = 'output/temp/'  # path of saving each stacking prediction
FEATURES_PATH = 'output/features/'  # path of dataset created in feat_verXX.py

# for saving the submitted format file(save_pred_as_submit_format())
SUBMIT_FORMAT = 'input/sample_submission.csv'

# check if path exsits
if not os.path.exists(DATA_PATH):
    print('making directory {}'.format(DATA_PATH))
    os.makedirs(DATA_PATH)

    print('making directory {}'.format(INPUT_PATH))
    os.makedirs(INPUT_PATH)

    print('making directory {}'.format(OUTPUT_PATH))
    os.makedirs(OUTPUT_PATH)

    print('making directory {}'.format(TEMP_PATH))
    os.makedirs(TEMP_PATH)

    print('making directory {}'.format(FEATURES_PATH))
    os.makedirs(FEATURES_PATH)
