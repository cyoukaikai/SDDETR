######################################################
# file path operation
######################################################

import os
import re
import sys
import shutil


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def non_blank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def generate_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_dir_for_file_if_not_exist(file_path):
    generate_dir_if_not_exist(os.path.dirname(os.path.abspath(file_path)))


def rmdir_if_exist(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)


def generate_dir_list_if_not_exist(dir_list):
    for dir_name in dir_list:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


def time_stamp_str():
    """
    >> print(time.strftime("%Y-%m-%d-%H-%M-%S"))
    2019-11-18-20-49-22
    :return:
    """
    import time
    return time.strftime("%Y-%m-%d-%H-%M-%S")


def assert_dir_exist(dir_name):
    if not os.path.isdir(dir_name):
        print(f'Directory {os.path.abspath(dir_name)} not exist, please check ...')
        sys.exit(0)


def assert_file_exist(file_path):
    if not os.path.isfile(file_path):
        print(f'File {os.path.abspath(file_path)} not exist, please check ...')
        sys.exit(0)
######################################################
# get file list or dir list
######################################################


def get_dir_list_recursively(walk_dir):
    """
    for root, subdirs, files in os.walk(rootdir):
    root: Current path which is "walked through"
    subdirs: Files in root of type directory
    files: Files in root (not in subdirs) of type other than directory

    dir_list.append( subdir ) does not make sense
    e.g., tests/test1 tests/test1/test2
    if dir_list.append( subdir ) will return tests, test1, test2

    , only_local_name=False
    """
    assert os.path.isdir(walk_dir)

    dir_list = []
    # https://www.mkyong.com/python/python-how-to-list-all-files-in-a-directory/
    for root, subdirs, files in os.walk(walk_dir):
        for subdir in subdirs:
            # if only_local_name:
            #     dir_list.append(subdir)
            # else:
            dir_list.append(os.path.join(root, subdir))
    return dir_list


def get_file_list_recursively(root_dir, ext_str=''):
    """
    ext_str = None (not specified, then files)
    ext_str: suffix for file, '.jpg', '.txt'  , only_local_name=False
    """
    assert_dir_exist(root_dir)

    file_list = []  # the full relative path from root_dir

    for root, subdirs, files in os.walk(root_dir):
        # print(subdirs)
        # print(files) # all files without dir name are saved

        for filename in files:
            if ext_str in filename:
                # if only_local_name:
                #     file_list.append(filename)
                # else:
                file_list.append(os.path.join(root, filename))
    # if not sort, then the images are not ordered.
    # 'visualization/image/3473/0257.jpg',
    #  'visualization/image/3473/0198.jpg',
    # 'visualization/image/3473/0182.jpg',
    # 'visualization/image/3473/0204.jpg'
    # file_list.sort()
    file_list = sorted(file_list, key=natural_sort_key)
    return file_list


def get_relative_file_list_recursively(root_dir, ext_str=''):
    file_list = get_file_list_recursively(root_dir, ext_str=ext_str)
    if len(file_list) > 0:
        for k, file in enumerate(file_list):
            file_list[k] = extract_relative_file_path(file, root_dir)
    return file_list


def extract_relative_file_path(file_path, root_dir):
    str_id = file_path.find(root_dir) + len(root_dir) + 1
    return file_path[str_id:]


def get_dir_list_in_directory(directory_path, only_local_name=True):
    """
    list all the directories under given 'directory_path'
    return a list of full path dir, in terms of
            directory_path  + sub_dir_name

    e.g.,
        get_dir_list_in_directory('truck_images')
        return
            ['truck_images/1', 'truck_images/2', ... ]
    """
    assert os.path.isdir(directory_path), f'Do not exist [{os.path.abspath(directory_path)}] ...'

    dir_path_list = []
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)  # images/2
        if os.path.isdir(f_path):
            if only_local_name:
                dir_path_list.append(f)
            else:
                dir_path_list.append(f_path)
    return dir_path_list


def get_file_list_in_directory(
        directory_path, only_local_name=False, ext_str=''
):
    # print(f'only_local_name = {only_local_name}, ext_str = {ext_str} ...')
    assert_dir_exist(directory_path)

    file_path_list = []
    # load image list
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)
        if os.path.isdir(f_path) or f.find(ext_str) == -1:
            # skip directories
            continue
        else:
            if only_local_name:
                file_path_list.append(f)
            else:
                file_path_list.append(f_path)
    return file_path_list
    # return directory_path


def get_json_file_list(directory_path, only_local_name=False):
    return get_file_list_in_directory(
        directory_path, only_local_name=only_local_name, ext_str='.json')


#################################
# path operation
######################################


def replace_same_level_dir(reference_dir_path, target_dir_path):
    """
    replace the last level dir path of "reference_dir_path" with "target_dir_path",
     and generate the "target_dir_path" in the same level with "reference_dir_path"
    :param reference_dir_path: e.g., images
    :param target_dir_path: e.g., labels
    :return:
    """
    return os.path.join(os.path.dirname(reference_dir_path), target_dir_path)


def file_path_last_two_level(file_path):
    file_names = file_path.split(os.path.sep)
    assert len(file_names) >= 2
    return os.path.join(file_names[-2], file_names[-1])


def extract_last_file_name(file_path):
    return file_path.split(os.path.sep)[-1]


def get_image_or_annotation_path(oldFilename, oldDir, newDir, newExt):
    old_path = oldFilename.replace(oldDir, newDir, 1)
    _, oldExt = os.path.splitext(old_path)
    new_path = old_path.replace(oldExt, newExt, 1)
    return new_path


def append_suffix_to_file_path(old_path, suffix):
    _, oldExt = os.path.splitext(old_path)
    newExt = suffix + oldExt
    new_path = old_path.replace(oldExt, newExt, 1)
    return new_path


def append_prefix_to_file_path(old_path, prefix):
    dir_name = os.path.dirname(old_path)
    basename = os.path.basename(old_path)
    new_path = os.path.join(
        dir_name, prefix + basename
    )
    return new_path


def replace_ext_str(old_path, new_ext_str):
    filename, oldExt = os.path.splitext(old_path)
    new_path = filename + new_ext_str
    return new_path


def get_basename_prefix(file_path):
    assert not os.path.isdir(file_path), \
        'Input should be a file not a directory'
    filename, _ = os.path.splitext(os.path.basename(file_path))
    return filename


def dir_name_up_n_levels(file_abspath, n):
    k = 0
    while k < n:
        file_abspath = os.path.dirname(file_abspath)
        k += 1

    return file_abspath


def specify_dir_list(data_root_dir, dir_list=None):
    if dir_list is None:
        dir_list = get_dir_list_in_directory(data_root_dir, only_local_name=True)
    return dir_list


def replace_dir_sep_with_underscore(dir_name):
    # used to save video for object_tracking result
    return dir_name.replace(os.path.sep, '_')


def generate_empty_file(file_path):
    try:
        open(file_path, 'w').close()
    except FileNotFoundError:
        print(f'file_path = {file_path} cannot be created.')
