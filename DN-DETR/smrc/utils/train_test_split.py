import os
import random

from .file_path import get_dir_list_in_directory
from .load_save import save_1d_list_to_file


def train_test_split_video_list(video_list, num_test, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    random.shuffle(video_list)
    test_video_list, train_video_list = video_list[:num_test], video_list[num_test:]
    return train_video_list, test_video_list


def generate_train_test_video_list(label_dir, num_test, random_seed=None):
    dir_list = get_dir_list_in_directory(label_dir)

    train_video_list, test_video_list = train_test_split_video_list(dir_list, num_test, random_seed)
    save_train_test_video_list(label_dir, train_video_list, test_video_list)
    return train_video_list, test_video_list


def save_train_test_video_list(label_dir, train_video_list, test_video_list, prefix=''):
    train_list_filename = os.path.join(
        os.path.dirname(label_dir),
        f'{prefix}train_{len(train_video_list)}video_list'
    )
    test_list_filename = os.path.join(
        os.path.dirname(label_dir),
        f'{prefix}test_{len(test_video_list)}video_list'
    )

    save_1d_list_to_file(train_list_filename, train_video_list)
    save_1d_list_to_file(test_list_filename, test_video_list)
    return train_video_list, test_video_list
