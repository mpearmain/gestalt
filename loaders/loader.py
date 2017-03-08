import pandas as pd
from sklearn.datasets import load_svmlight_file

# feature list
def load_pandas(flist):
    """
    Usage: set train, target, and test key and feature files,
    pre-processing should be done so that there are not duplicate features in different datasets.

    :param flist: files list
    :example:

    FEATURE_LIST_stage2 = {
                'train':(TEMP_PATH + 'v1_stage1_all_fold.csv',
                         TEMP_PATH + 'v2_stage1_all_fold.csv',
                         TEMP_PATH + 'v3_stage1_all_fold.csv',), #target is not in 'train'
                'target':(INPUT_PATH + 'target.csv',), #target is in 'target'
                'test':(TEMP_PATH + 'v1_stage1_test.csv',
                         TEMP_PATH + 'v2_stage1_test.csv',
                         TEMP_PATH + 'v3_stage1_test.csv',),
                }
    """
    if (len(flist['train']) == 0) or (len(flist['target']) == 0) or (len(flist['test']) == 0):
        raise Exception('Train, Target, and Test must be set at least one file, respectively.')

    x_train = pd.DataFrame()
    test = pd.DataFrame()

    print('Reading train dataset')
    for i in flist['train']:
        x_train = pd.concat([x_train, pd.read_csv(i, index_col=0)], axis=1)
        print('train dataset is created')

    print('Reading target data')
    y_train = pd.read_csv(flist['target'][0], index_col=0)

    print('Reading train dataset')
    for i in flist['test']:
        test = pd.concat([test, pd.read_csv(i, index_col=0)], axis=1)

    assert (all(x_train.columns == test.columns))
    print('Train and test cols align, lets get modelling')
    return x_train, y_train, test


def load_libsvm(flist):
    """
    Usage: set train, target, and test key and feature files,
    pre-processing should be done so that there are not duplicate features in different datasets.

    :param flist: files list
    :example:

    FEATURE_LIST_stage2 = {
                'train':(TEMP_PATH + 'v1_stage1_all_fold.csv',
                         TEMP_PATH + 'v2_stage1_all_fold.csv',
                         TEMP_PATH + 'v3_stage1_all_fold.csv',), #target is not in 'train'
                'target':(INPUT_PATH + 'target.csv',), #target is in 'target'
                'test':(TEMP_PATH + 'v1_stage1_test.csv',
                         TEMP_PATH + 'v2_stage1_test.csv',
                         TEMP_PATH + 'v3_stage1_test.csv',),
                }
    """
    pass
    # if (len(flist['train']) == 0) or (len(flist['target']) == 0) or (len(flist['test']) == 0):
    #     raise Exception('Train, Target, and Test must be set at least one file, respectively.')
    #
    #
    # print('Reading train dataset')
    # X_train, y_train, X_test, y_test = load_svmlight_files(
    #     ...("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
    #     print('train dataset is created')
    #
    # print('Reading train dataset')
    # for i in flist['test']:
    #     X_train, y_train, X_test, y_test = load_svmlight_files(
    #         ...("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
    #
    # assert (all(x_train.shape[1] == test.columns.shape[1]))
    # print('Train and test cols align, lets get modelling')
    # return x_train, y_train, test


def load_libpd(flist):
    """
    Usage: set train, target, and test key and feature files,
    pre-processing should be done so that there are not duplicate features in different datasets.

    :param flist: files list
    :example:

    FEATURE_LIST_stage2 = {
                'train_pd':(TEMP_PATH + 'v1_stage1_all_fold.csv',
                            TEMP_PATH + 'v2_stage1_all_fold.csv',
                            TEMP_PATH + 'v3_stage1_all_fold.csv',), #target is not in 'train'
                'train_libsvm':(TEMP_PATH + 'v1_stage1_all_fold.csv',
                                TEMP_PATH + 'v2_stage1_all_fold.csv',
                                TEMP_PATH + 'v3_stage1_all_fold.csv',),
                'target':(INPUT_PATH + 'target.csv',), #target is in 'target'
                'test_pd':(TEMP_PATH + 'v1_stage1_test.csv',
                           TEMP_PATH + 'v2_stage1_test.csv',
                           TEMP_PATH + 'v3_stage1_test.csv',),
                'test_libsvm':(TEMP_PATH + 'v1_stage1_test.csv',
                               TEMP_PATH + 'v2_stage1_test.csv',
                               TEMP_PATH + 'v3_stage1_test.csv',),
                }
    """
    pass
    # if (len(flist['train']) == 0) or (len(flist['target']) == 0) or (len(flist['test']) == 0):
    #     raise Exception('Train, Target, and Test must be set at least one file, respectively.')
    #
    # x_train = pd.DataFrame()
    # test = pd.DataFrame()
    #
    # print('Reading train dataset')
    # for i in flist['train']:
    #     x_train = pd.concat([x_train, pd.read_csv(i, index_col=0)], axis=1)
    #     print('train dataset is created')
    #
    # print('Reading target data')
    # y_train = pd.read_csv(flist['target'][0], index_col=0)
    #
    # print('Reading train dataset')
    # for i in flist['test']:
    #     test = pd.concat([test, pd.read_csv(i, index_col=0)], axis=1)
    #
    # assert (all(x_train.columns == test.columns))
    # print('Train and test cols align, lets get modelling')
    # return x_train, y_train, test