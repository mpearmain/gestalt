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
    # train_test_cv1_sparse = sparse.hstack((train_test, c_vect_sparse_1)).tocsr()
    # something like this
    # x_train = train_test_cv1_sparse[:ntrain, :]
    # x_test = train_test_cv1_sparse[ntrain:, :]
    # features += c_vect_sparse1_cols
    # return x_train, y_train, test
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
    # train_test_cv1_sparse = sparse.hstack((train_test, c_vect_sparse_1)).tocsr()
    # something like this
    # x_train = train_test_cv1_sparse[:ntrain, :]
    # x_test = train_test_cv1_sparse[ntrain:, :]
    # features += c_vect_sparse1_cols
    # return x_train, y_train, test