import numpy as np


def compute_l2_dist(list1, list2):
    """
    L2 norm, np.linalg.norm([3,4]) = 5.0
    :param list2: 1d list
    :param list1: 1d list
    :return:
    """
    return np.linalg.norm(
        np.asarray(list1) - np.asarray(list2)
    )
