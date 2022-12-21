import shutil
from tqdm import tqdm
import os
import cv2

from .file_path import get_file_list_recursively, get_dir_list_in_directory, get_file_list_in_directory, \
    get_image_or_annotation_path, replace_dir_sep_with_underscore, \
    generate_dir_if_not_exist, time_stamp_str
from .load_save import load_multi_column_list_from_file, load_1d_list_from_file, \
    save_1d_list_to_file, save_1d_list_to_file_incrementally
from .image_video import get_image_file_list_in_directory, \
    get_image_size
from .bbox import empty_annotation_file
from .base import diff_list


def find_dir_diff(root_dir1, root_dir2):
    dir_list1 = get_dir_list_in_directory(root_dir1, only_local_name=True)
    dir_list2 = get_dir_list_in_directory(root_dir2, only_local_name=True)
    unique_list1, unique_list2 = diff_list(dir_list1, dir_list2)
    return unique_list1 + unique_list2

#########################################################
# yolo-v4, data clean
############################################################


def remove_txt_files(image_root_dir):
    """
    :param image_root_dir:
    :return:
    """
    print(f'To remove all the txt files in {image_root_dir} ...')
    count = 0
    dir_list = get_dir_list_in_directory(image_root_dir)

    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To remove all the txt files in {image_root_dir}: '
                             f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(image_root_dir, dir_name), ext_str='.txt'
        )
        for txt_file in txt_file_list:
            os.remove(txt_file)
            count += 1
    print(f'Total {count} files removed ...')


def remove_dirs(label_root_dir, dir_list):
    print(f'Remove {len(dir_list)} dirs from {label_root_dir} ...')
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        source_dir = os.path.join(label_root_dir, dir_name)
        if os.path.isdir(source_dir):
            num_txt_files = len(get_file_list_in_directory(source_dir))
            pbar.set_description(f'| Remove {source_dir}, {num_txt_files} files, '
                                 f'({dir_idx}/{len(dir_list)}) ...')
            # remove empty dir
            shutil.rmtree(source_dir)
            # os.rmdir(source_dir)


def copy_sub_dirs(source_data_dir, target_data_dir, dir_list=None, skip_if_not_exist=False):
    """
    Only for copying sub directories, files under the source_data_dir are not copied.
    :param dir_list:
    :param source_data_dir:
    :param target_data_dir:
    :return:
    """
    print(f'To copy directories from {source_data_dir} to {target_data_dir} ...')
    if dir_list is None:
        dir_list = get_dir_list_in_directory(source_data_dir)

    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To copy directory from {source_data_dir} to {target_data_dir} :'
                             f'{dir_name} ({dir_idx}/{len(dir_list)}) ...')
        source_sub_dir = os.path.join(source_data_dir, dir_name)
        tar_sub_dir = os.path.join(target_data_dir, dir_name)

        if not os.path.isdir(source_sub_dir):
            if skip_if_not_exist:
                # skip the copy if the dir name not exist in the source root dir
                continue
            else:
                print(f'{source_sub_dir} does not exists')
                raise FileNotFoundError

        if os.path.isdir(tar_sub_dir):  # if tar_sub_dir already exists in the target location
            # skip the copied sub dirs to avoid redo the coping for saving the time
            copied_file_list = get_file_list_in_directory(tar_sub_dir)
            if len(copied_file_list) > 0:
                if len(copied_file_list) == len(get_file_list_in_directory(source_sub_dir)):
                    continue
            # We must remove the tar_sub_dir as shutil.copytree will call os.makedirs(dst)
            shutil.rmtree(tar_sub_dir)

        shutil.copytree(
            source_sub_dir,
            tar_sub_dir
        )
    print(f'Total {len(dir_list)} dir_list copied ...')


def move_sub_dirs(source_data_dir, target_data_dir, dir_list=None, skip_if_not_exist=False):
    """
    Only for copying sub directories, files under the source_data_dir are not copied.
    :param skip_if_not_exist:
    :param dir_list:
    :param source_data_dir:
    :param target_data_dir:
    :return:
    """
    print(f'To move directories from {source_data_dir} to {target_data_dir} ...')
    if dir_list is None:
        dir_list = get_dir_list_in_directory(source_data_dir)

    pbar = tqdm(enumerate(dir_list))
    count = 1
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To copy directory from {source_data_dir} to {target_data_dir} :'
                             f'{dir_name} ({dir_idx}/{len(dir_list)}) ...')
        source_sub_dir = os.path.join(source_data_dir, dir_name)
        tar_sub_dir = os.path.join(target_data_dir, dir_name)

        if not os.path.isdir(source_sub_dir):
            if skip_if_not_exist:
                # skip the copy if the dir name not exist in the source root dir
                continue
            else:
                print(f'{source_sub_dir} does not exists')
                raise FileNotFoundError

        if os.path.isdir(tar_sub_dir):  # if tar_sub_dir already exists in the target location
            # skip the copied sub dirs to avoid redo the coping for saving the time
            copied_file_list = get_file_list_in_directory(tar_sub_dir)
            if len(copied_file_list) > 0:
                if len(copied_file_list) == len(get_file_list_in_directory(source_sub_dir)):
                    continue
            # We must remove the tar_sub_dir as shutil.copytree will call os.makedirs(dst)
            shutil.rmtree(tar_sub_dir)
        count += 1
        shutil.move(source_sub_dir, tar_sub_dir)

    print(f'Total {count} of {len(dir_list)} dir_list moved from {source_data_dir} to {target_data_dir}  ...')


def copy_labels(source_label_dir, target_label_dir):
    """
    Only for copying txt files.
    :param source_label_dir:
    :param target_label_dir:
    :return:
    """
    print(f'To copy directory from {source_label_dir} to {target_label_dir} ...')
    dir_list = get_dir_list_in_directory(source_label_dir)
    count = 0
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To copy directory from {source_label_dir} to {target_label_dir} :'
                             f'{dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(source_label_dir, dir_name),
            ext_str='.txt', only_local_name=False
        )
        for txt_file in txt_file_list:
            target_file = txt_file.replace(
                source_label_dir, target_label_dir, 1
            )
            # print(f'Copying {txt_file} to {target_file} ...')
            shutil.copyfile(txt_file, target_file)
            count += 1
    print(f'Total {count} files copied ...')


def count_empty_txt_file(label_root_dir):
    """
    root_dir = os.path.join('datasets', 'DensoData_No_Masking8SampleVideo')
    parent_dir_to_process = os.path.join(root_dir, 'labels-first98videos') #tmp_YOLO_FORMAT3807

    label_root_dir = 'data/datasets/TruckDB_Face/all7594'
    :param label_root_dir:
    :return:
    """
    dir_list = get_dir_list_in_directory(label_root_dir)
    count = 0
    # dir_list = dir_list[0]
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name), ext_str='.txt'
        )
        for txt_file in txt_file_list:
            statinfo = os.stat(txt_file)
            # print(f'{txt_file}, {statinfo.st_size}')
            if statinfo.st_size == 0:
                print(f'{txt_file} empty ... ')
                count += 1
    print(f'Total {count} files empty ...')


def add_empty_txt_files(image_root_dir, label_root_dir=None):
    """Generating empty files for images without annotation (empty images).
    """
    # for yolo v4, the labels and images are in the same directory
    # for yolo v3, labels and images are in different directories.
    if label_root_dir is None:
        label_root_dir = image_root_dir
    print(f'To padding labels to {image_root_dir} based on txt files in {label_root_dir} ...')
    dir_list = get_dir_list_in_directory(image_root_dir)
    count = 0
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To padding labels to {image_root_dir}: {dir_name} [{dir_idx}/{len(dir_list)}] ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        image_path_list = get_image_file_list_in_directory(
            os.path.join(image_root_dir, dir_name)
        )
        for image_path in image_path_list:
            txt_file_name = get_image_or_annotation_path(
                image_path, image_root_dir, label_root_dir, '.txt'
            )
            if not os.path.isfile(txt_file_name):
                empty_annotation_file(txt_file_name)
                # print(f'Generating empty file {txt_file_name} ...')
                count += 1
    print(f'Total {count} empty files generated ...')


#########################################################
# resize images
#########################################################

def delete_empty_file(label_root_dir):
    """
    root_dir = os.path.join('datasets', 'DensoData_No_Masking8SampleVideo')
    parent_dir_to_process = os.path.join(root_dir, 'labels-first98videos') #tmp_YOLO_FORMAT3807

    label_root_dir = 'data/datasets/TruckDB_Face/all7594'
    :param label_root_dir:
    :return:
    """
    dir_list = get_dir_list_in_directory(label_root_dir)

    # dir_list = dir_list[0]
    for dir_idx, dir_name in enumerate(dir_list):
        print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name)
        )
        for txt_file in txt_file_list:
            statinfo = os.stat(txt_file)
            print(f'{txt_file}, {statinfo.st_size}')
            if statinfo.st_size == 0:
                print(f'Removing {txt_file} ... ')
                os.remove(txt_file)

        txt_file_list_remains = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name)
        )
        print(f'After processing {dir_name}, {len(txt_file_list_remains)} of {len(txt_file_list)} remains ')


def delete_empty_dirs(data_root_dir):
    dir_list = get_dir_list_in_directory(data_root_dir)
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        source_dir = os.path.join(data_root_dir, dir_name)

        num_files = len(get_file_list_in_directory(source_dir))
        if num_files == 0:
            # remove empty dir
            pbar.set_description(f'| Remove {source_dir},  ({dir_idx}/{len(dir_list)}) ...')
            os.rmdir(source_dir)


def delete_empty_files_and_dirs(data_root_dir):
    # if dir_list is None:  #  dir_list=None
    #     dir_list = get_dir_list_in_directory(
    #         data_root_dir
    #     )
    delete_empty_file(data_root_dir)
    delete_empty_dirs(data_root_dir)


def resize_image_pairwise_comparison(src_image_root_dir, refer_image_root_dir):
    """
    Resize the images in src_image_root_dir based on the size of the corresponding images in refer_image_root_dir
    :param src_image_root_dir:
    :param refer_image_root_dir:
    :return:
    """
    assert len(find_dir_diff(src_image_root_dir, refer_image_root_dir)) == 0
    target_image_root_dir = src_image_root_dir + '_resized'
    generate_dir_if_not_exist(target_image_root_dir)
    print(f'To resize images in {src_image_root_dir} based on the corresponding images in '
          f'{refer_image_root_dir}, results are saved in  {target_image_root_dir} ...')

    dir_list = get_dir_list_in_directory(src_image_root_dir)
    count = 0
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'To process {dir_name} [{dir_idx}/{len(dir_list)}] ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        image_path_list = get_image_file_list_in_directory(
            os.path.join(src_image_root_dir, dir_name)
        )
        target_dir_path = os.path.join(target_image_root_dir, dir_name)
        generate_dir_if_not_exist(target_dir_path)
        for image_path in image_path_list:
            refer_image_path = image_path.replace(src_image_root_dir, refer_image_root_dir, 1)
            h, w = get_image_size(refer_image_path)
            resized_image_path = image_path.replace(src_image_root_dir, target_image_root_dir, 1)

            tmp_img = cv2.imread(image_path)
            resized_img = cv2.resize(tmp_img, (w, h))
            cv2.imwrite(resized_image_path, resized_img)
            # print(f'Generating empty file {txt_file_name} ...')
            count += 1
    print(f'Total {count} images resized ...')


def report_image_size(image_dir):
    """
    # report the statistics for the image information, e.g.,
    image_dir = 'data/driver_face/test_data/sample91_images'
    :param image_dir:
    :return:
    """
    image_list_dir = image_dir + '_image_list'
    generate_dir_if_not_exist(image_list_dir)
    dir_list = get_dir_list_in_directory(image_dir)

    video_image_size = []
    image_dir_for_detection = []
    for idx, dir_name in enumerate(dir_list):
        # print(f'Processing {dir_name} [{idx + 1}/{len(dir_list)}] ... ')
        # image_file_list = get_file_list_recursively(
        #     os.path.join(image_dir, dir_name)
        # )
        image_file_list = get_image_file_list_in_directory(
            os.path.join(image_dir, dir_name)
        )
        if len(image_file_list) == 0:
            print(f'{dir_name} has no images...')
            continue

        image_name = image_file_list[0]
        img = cv2.imread(image_name)
        if img is not None:
            height, width, _ = img.shape
            video_infor = ','.join(map(str, [dir_name, len(image_file_list), width, height]))
        else:
            video_infor = ','.join(map(str, [dir_name, len(image_file_list), 0, 0]))

        print(f'{video_infor}')
        video_image_size.append(video_infor)

        image_list_abs_path = [os.path.abspath(x) for x in image_file_list]
        file_name = os.path.join(image_list_dir, dir_name + '.txt')
        save_1d_list_to_file(file_name, image_list_abs_path)

        image_dir_for_detection.append(os.path.abspath(file_name))

    save_1d_list_to_file(image_dir + '_size_infor.txt', video_image_size)
    save_1d_list_to_file(os.path.join(image_list_dir, 'video_infor.txt'), image_dir_for_detection)
    # smrc.not_used.save_1d_list_to_file(
    #   os.path.join('sample91_images', 'video_infor.txt'),image_dir_for_detection
    # )


def count_num_lines(label_root_dir, ext_str='.txt'):
    """
    Count the number of lines in txt files.
    :param ext_str:
    :param label_root_dir:
    :return:
    """
    dir_list = get_dir_list_in_directory(label_root_dir)
    file_count = 0
    line_count = 0
    # dir_list = dir_list[0]
    pbar = tqdm(enumerate(dir_list))
    for dir_idx, dir_name in pbar:
        pbar.set_description(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        # print(f'Processing {dir_name} ({dir_idx}/{len(dir_list)}) ...')
        txt_file_list = get_file_list_in_directory(
            os.path.join(label_root_dir, dir_name), ext_str=ext_str
        )
        for txt_file in txt_file_list:
            statinfo = os.stat(txt_file)
            # print(f'{txt_file}, {statinfo.st_size}')
            if statinfo.st_size > 0:
                ann_list = load_multi_column_list_from_file(txt_file)
                # print(f'{dir_name} {len(ann_list)}  ... ')
                file_count += 1
                line_count += len(ann_list)
    print(f'Total {file_count} non empty file, {line_count} lines ...')


#############################################
#
############################################

def move_and_rename_files_recursively(dir_path):
    """
    Move and rename the files in dir_path with depth > 1 to
    files with depth 1.
    For instance, 'test_data/0000.jpg', 'test_data/1/0000.jpg'
    will become
        'test_data/0000.jpg', 'test_data/1_0000.jpg'
    respectively
    :param dir_path:
    :return:
    """
    sub_dir_list = get_dir_list_in_directory(dir_path)
    # if there is no sub_dir_list
    if len(sub_dir_list) == 0:
        return
    else:
        file_list = []
        for sub_dir_name in sub_dir_list:
            file_list += get_file_list_recursively(
                    os.path.join(dir_path, sub_dir_name)
                )

        print(f'Total {len(file_list)} files need to be renamed ...')
        pbar = tqdm(file_list)
        for file in pbar:
            pbar.set_description(f'Processing {file} ...')
            str_id = file.find(dir_path) + len(dir_path) + 1
            new_basename = replace_dir_sep_with_underscore(
                file[str_id:]
            )
            os.rename(file, os.path.join(dir_path, new_basename))

        for sub_dir_name in sub_dir_list:
            shutil.rmtree(os.path.join(dir_path, sub_dir_name))


def multi_level_dir_to_two_level(data_root_dir):
    """
    Transfer a multi-level dir a two levels dir.
    This will be useful for transferring images or labels with multiple
    depth to 2-level depth for easy use.
    :param data_root_dir:
    :return:
    """

    dir_list = get_dir_list_in_directory(data_root_dir)
    for k, dir_name in enumerate(dir_list):
        print(f'Processing {k}/{len(dir_list)} dir {dir_name}')

        # move all the files in the dir_name to one level dir
        target_dir = os.path.join(data_root_dir, dir_name)
        move_and_rename_files_recursively(target_dir)

        # sub_dir_list = get_dir_list_in_directory(
        #     target_dir, only_local_name=False
        # )
        # for sub_dir in sub_dir_list:
        #     move_and_rename_files_recursively(sub_dir)


# def report_image_size_complete(image_dir):
#     """
#     # report the statistics for the image information, e.g.,
#     image_dir = 'data/driver_face/test_data/sample91_images'
#     :param image_dir:
#     :return:
#     """
#     image_list_dir = image_dir + '_image_list'
#     generate_dir_if_not_exist(image_list_dir)
#     dir_list = get_dir_list_in_directory(image_dir)
#
#     video_image_size = []
#     image_dir_for_detection = []
#     for idx, dir_name in enumerate(dir_list):
#         # print(f'Processing {dir_name} [{idx + 1}/{len(dir_list)}] ... ')
#         image_file_list = get_file_list_recursively(
#             os.path.join(image_dir, dir_name)
#         )
#         if len(image_file_list) == 0:
#             print(f'{dir_name} has no images...')
#             continue
#
#         image_name = image_file_list[0]
#         img = cv2.imread(image_name)
#         if img is not None:
#             height, width, _ = img.shape
#             video_infor = ','.join(map(str, [dir_name, len(image_file_list), width, height]))
#         else:
#             video_infor = ','.join(map(str, [dir_name,len(image_file_list), 0, 0]))
#
#         print(f'{video_infor}')
#         video_image_size.append(video_infor)
#
#         image_list_abs_path = [os.path.abspath(x) for x in image_file_list]
#         file_name = os.path.join(image_list_dir, dir_name + '.txt')
#         save_1d_list_to_file(file_name, image_list_abs_path)
#
#         image_dir_for_detection.append(os.path.abspath(file_name))
#
#     save_1d_list_to_file(image_dir + '_size_infor.txt', video_image_size)
#     save_1d_list_to_file(os.path.join(image_list_dir, 'video_infor.txt'), image_dir_for_detection)
#     # smrc.not_used.save_1d_list_to_file(
#     #   os.path.join('sample91_images', 'video_infor.txt'),image_dir_for_detection
#     # )1


def generate_file_for_image_list(
        image_root_dir, result_file, dir_list=None, ext_str=None):
    if dir_list is None:
        dir_list = get_dir_list_in_directory(image_root_dir)
    assert len(dir_list) > 0
    # print(f'| Total {len(dir_list)} directories. ')

    pbar = tqdm(enumerate(dir_list))
    count = 0
    for k, dir_name in pbar:
        pbar.set_description(f'Loading images for {dir_name}, Total {count} loaded [{k}/{len(dir_list)}]')
        if ext_str is not None:
            image_path_list = get_file_list_in_directory(
                os.path.join(image_root_dir, dir_name), ext_str=ext_str
            )
        else:
            image_path_list = get_image_file_list_in_directory(
                os.path.join(image_root_dir, dir_name)
            )
        count += len(image_path_list)
        save_1d_list_to_file_incrementally(
            file_path=result_file, list_to_save=image_path_list
        )


def load_image_list(image_root_dir, dir_list=None, ext_str=None):
    tmp_file = 'TMP___image_file_list_' + time_stamp_str()
    generate_file_for_image_list(
        image_root_dir=image_root_dir,
        dir_list=dir_list, result_file=tmp_file, ext_str=ext_str)
    image_path_list = load_1d_list_from_file(tmp_file)

    # delete the deprecated file
    os.remove(tmp_file)
    return image_path_list

# def load_image_list_slow(image_root_dir, dir_list=None):
#     if dir_list is None:
#         dir_list = get_dir_list_in_directory(image_root_dir)
#     assert len(dir_list) > 0
#     # print(f'| Total {len(dir_list)} directories. ')
#     image_path_list = []
#     pbar = tqdm(enumerate(dir_list))
#     for k, dir_name in pbar:
#         pbar.set_description(f'Loading images for {dir_name} [{k}/{len(dir_list)}]')
#         image_path_list += get_image_file_list_in_directory(
#             os.path.join(image_root_dir, dir_name)
#         )
#     return image_path_list
