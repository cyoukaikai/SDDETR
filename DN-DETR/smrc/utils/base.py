import collections

from .load_save import load_1d_list_from_file, save_1d_list_to_file
from .file_path import assert_file_exist


def flattened_2d_list(my_list):
    return [x for sublist in my_list for x in sublist]


def unique_element_in_2d_list(my_list):
    return [list(t) for t in set(tuple(element) for element in my_list)]


def remove_one_item_from_1d_list_file(list1d_file, item, assert_none_exist=False):
    """
    Remove one element from a 1 d list file. This function can be used to move one directory name from
    a file which consists of a list of directories.
    :param list1d_file:
    :param item:
    :param assert_none_exist:
    :return:
    """
    assert_file_exist(list1d_file)

    my_list = load_1d_list_from_file(list1d_file)
    # remove -> removes the first matching value
    if item in my_list:
        my_list.remove(item)
        save_1d_list_to_file(file_path=list1d_file, list_to_save=my_list)
    else:
        print(f'Item {item} not in {list1d_file}')
        if assert_none_exist:
            raise AssertionError
    return my_list

###########################################
# other often used operations in our project
############################################

def repeated_element_in_list(list_to_to_check):
    return [item for item, count in collections.Counter(list_to_to_check).items()
            if count > 1]


def repeated_element_and_count_in_list(list_to_to_check):
    return [(item, count) for item, count in collections.Counter(list_to_to_check).items()
            if count > 1]


def shared_list(list1, list2):
    return repeated_element_in_list(list1 + list2)


def diff_list(list1, list2):
    unique_list1 = [x for x in list1 if x not in list2]
    unique_list2 = [x for x in list2 if x not in list1]
    print(f'len(set(list1)) = {len(set(list1))}, len(set(list2)) = {len(set(list2))}')
    print(f'Repeated element in list1 {repeated_element_and_count_in_list(list1)} ... ')
    print(f'Repeated element in list2 {repeated_element_and_count_in_list(list2)} ... ')
    print("Difference: Only in list1 \n", set(list1) - set(list2))
    print("Difference: Only in list2 \n", set(list2) - set(list1))

    return unique_list1, unique_list2


def exclude_one_list_from_another(full_dir_list, dir_list_to_exclude):
    remaining_video_list = []
    for video in full_dir_list:
        if video not in dir_list_to_exclude:
            remaining_video_list.append(video)

    return remaining_video_list


def exclude_one_file_list_from_another_file(
        full_dir_list_file, dir_list_to_exclude_file,
        resulting_file=None
):

    video_list1 = load_1d_list_from_file(full_dir_list_file)

    video_list2 = load_1d_list_from_file(dir_list_to_exclude_file)

    remaining_video_list = exclude_one_list_from_another(video_list1, video_list2)

    if resulting_file is not None:
        save_1d_list_to_file(resulting_file, remaining_video_list)

    return remaining_video_list


def float_to_str(v, num_digits):
    result_format = f"%.{num_digits}f"
    return f'{result_format % v}'


def int_to_str(v, num_digits):
    result_format = f"%0{num_digits}d"
    return f'{result_format % v}'


def display_first_three_item_in_list(my_list):
    print(f'Printing the first element in a list, total {len(my_list)} ')
    for k, ele in enumerate(my_list):
        print(f'K = {k}, element: {ele}')
        if k == 2: break
