# coding=utf-8
import os
from models.base import DATA_PATH, INPUT_PATH, OUTPUT_PATH, TEMP_PATH, FEATURES_PATH

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
