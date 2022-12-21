import random
import numpy as np


def sample_item_in_list_evenly(my_list, num_sample):
    """
    Sampling sample_num items from my_list evenly based on the item id.
    """
    assert num_sample <= len(my_list)

    step = int(np.floor(len(my_list) / num_sample))
    inds = list(range(0, len(my_list), step))[:num_sample]
    assert len(inds) == num_sample
    result = [my_list[k] for k in inds]
    return result, inds


def sample_item_in_list_randomly(my_list, num_sample):
    return random.choices(my_list, k=num_sample)