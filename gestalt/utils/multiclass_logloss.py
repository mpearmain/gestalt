import numpy as np


def mlogloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    :param eps: cutoff tolerence values (REMEMBER log(0) is not defined mathematically)
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
