import json
import os
######################################################
# handle the result of yolo based object_detection
######################################################

import imagesize
import numpy as np

from .detection_process import nms_detection_dict, \
    filter_out_low_score_detection_dict, det_dict_to_smrc_tracking_format
from . import image_path_last_two_level
from .. import get_json_file_list, natural_sort_key, \
    generate_dir_if_not_exist, get_file_list_in_directory, \
    replace_ext_str, save_bbox_to_file

# class YOLODetFormat:
#     V3 = 0  # bbox format: x1, y1, x2, y2
#     V4 = 1  # bbox format: normalized x1, y1, x2, y2, i.e.,
#     # x1, x2 / img_width, y1, y2 / img_height
#     #'yolov4', 'yolov3'


#############################################
# base functions
#############################################

def extract_smrc_json_dir(json_file_name):
    return json_file_name.replace('.json', '').replace('smrc_', '')


def get_json_dir_list(directory_path):
    json_files = get_json_file_list(directory_path, only_local_name=True)
    dir_list = [extract_smrc_json_dir(x) for x in json_files]
    return dir_list


def get_smrc_json_file_list(directory_path, only_local_name=False):
    # print(directory_path)
    file_path_list = []
    # load image list
    for f in sorted(os.listdir(directory_path), key=natural_sort_key):
        f_path = os.path.join(directory_path, f)
        # print(f_path)
        # print(f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        elif os.path.isfile(f_path) and f.find('.json') > 0:
            # and f.startswith('smrc_')
            # f.find('smrc_') >= 0 also ok if not use ( f.startswith('smrc_') )
            if only_local_name:
                file_path_list.append(f)
            else:
                file_path_list.append(f_path)
    print(f'{len(file_path_list)} json files loaded with the format *.json.')
    return file_path_list


def yolo_bbox_rect_to_smrc_rect(bbox_rect, img_h, img_w):
    center_x, center_y, width, height = bbox_rect
    x1 = round((center_x - 0.5 * width) * img_w)
    x2 = round((center_x + 0.5 * width) * img_w)
    y1 = round((center_y - 0.5 * height) * img_h)
    y2 = round((center_y + 0.5 * height) * img_h)
    if x1 < 0: x1 = 0
    if x2 > img_w: x2 = img_w - 1
    if y1 < 0: y1 = 0
    if y2 > img_h: y2 = img_h - 1
    return [x1, y1, x2, y2]


def det_list_to_bbox_list(image_det):
    """
    For smrc format det_list
    :param image_det:
    :return:
    """
    return [det[:5] for det in image_det]


########################################
# for yolov3 detection
####################################

def load_yolov3_json_detection_to_dict(
        json_detection_file, detection_dict=None,
        score_thd=None, nms_thd=None,
        short_image_path=True
):
    """Load the YOLO detections that are saved in json format to object_detection dict.

    :param json_detection_file: one file include the detections of one video, with the
        following format,
        [
            {"image_path":"/home/smrc/darknet/Detection_Taxi-raw-data_20200427_add2/62054/0000.jpg",
            "category_id":1, "bbox":[359, 220, 390, 236], "score":0.942105},
            ...
        ]
    :param detection_dict: if not none, continue to add the object_detection to the dict, this
        is useful for ensemble of multiple detections
    :param short_image_path: if true, only save the last two levels of the image path,
        i.e., 3440/0000.jpg
    :param score_thd: remove the low score object_detection, score < score_thd
    :param nms_thd: non maximum suppression threshold
    :return:
        object_detection dict with the format of
            [class_idx, xmin, ymin, xmax, ymax, score]
        A lot of public codes use the format of [class_idx, score, xmin, ymin, xmax, ymax],
        If we do need the public format, just conduct transformation.
    """
    # {'image_path': '/home/smrc/Data/1000videos/3440/0000.jpg', 'category_id': 2,
    # 'bbox': [275, 187, 78, 53], 'score': 0.984684}
    # print(f'Loading {json_detection_file} to detection_list...')
    with open(json_detection_file) as json_file:
        json_detection_data = json.load(json_file)

    # all the detections regarding this directory are saved here
    if detection_dict is None:
        detection_dict = {}

    count = 0
    for detection in json_detection_data:  # detection_idx,
        # load the image name
        image_path = detection['image_path']
        if short_image_path:
            image_path = image_path_last_two_level(image_path)

        class_idx = detection['category_id']
        xmin, ymin, xmax, ymax = list(map(int, detection['bbox']))
        score = float(detection['score'])

        if score_thd is not None and score < score_thd:
            count += 1
            # print(f'    ignored object_detection {[class_idx, xmin, ymin, xmax, ymax, score]}...')
        else:
            if image_path in detection_dict:
                detection_dict[image_path].append(
                    [class_idx, xmin, ymin, xmax, ymax, score])
            else:
                detection_dict[image_path] = [
                    [class_idx, xmin, ymin, xmax, ymax, score]
                ]
    det_num = np.sum([len(dets) for key, dets in detection_dict.items()])
    print(f'    {count} object_detection ignored due to score_thd {score_thd}, '
          f'remaining {det_num} detections ...')

    if nms_thd is not None:
        detection_dict = nms_detection_dict(detection_dict, nms_thd)
    det_num_after_nms = np.sum([len(dets) for key, dets in detection_dict.items()])
    print(f'    {det_num - det_num_after_nms} object_detection ignored due to nms_thd {nms_thd}, '
          f'remaining {det_num_after_nms} detections ...')
    return detection_dict


# ===================================================
# ensemble detection
# ==================================================

def load_multiple_json_detection_to_dict(
        json_file_list, score_thd=None, nms_thd=None, short_image_path=True
):
    """# Ensemble multiple detection files
    """
    detection_dict = {}
    for json_file in json_file_list:
        detection_dict = load_yolov3_json_detection_to_dict(
            json_detection_file=json_file,
            detection_dict=detection_dict,
            score_thd=score_thd,
            nms_thd=nms_thd,  # conduct nms in separate files as it should be
            short_image_path=short_image_path
        )

    # conduct nms again after detections from all files are loaded
    # Note that unexpected behavior may occur if the detections are from different
    # detectors and the scores are not normalized.
    if nms_thd is not None:
        detection_dict = nms_detection_dict(detection_dict, nms_thd)

    return detection_dict


#################################
# load data for tracking
#################################

def load_json_det_files_to_tracking_format(
        json_file_list, test_image_list,
        score_thd=None, nms_thd=None
):
    """
    load object_detection from json file to detection_list with non maximum suppression
    This function can be used for processing the object_detection of a single video, or
    the detections of multiple video as long as all the images are included in
    test_image_list
    :param json_file_list:
    :param test_image_list:
    :param nms_thd:
    :param score_thd:
    :return:
    """
    detection_dict = load_multiple_json_detection_to_dict(
        json_file_list, score_thd=score_thd, nms_thd=nms_thd,
        short_image_path=True
    )

    detection_list = det_dict_to_smrc_tracking_format(
        detection_dict, test_image_list
    )
    return detection_list


def json_det_to_tracking_format(
        json_file, test_image_list, score_thd=None, nms_thd=None
):
    """
    [class_idx, x1, y1, x2, y2, score],
    :param json_file:
    :param test_image_list:
    :param nms_thd:
    :param score_thd:
    :return: a detection list of the format of [class_idx, x1, y1, x2, y2, score]
    """
    return load_json_det_files_to_tracking_format(
        json_file_list=[json_file],
        test_image_list=test_image_list,
        score_thd=score_thd,
        nms_thd=nms_thd
    )


########################################
# for yolov3 detection
####################################

def parse_yolov4_frame_det(frame_det, image_root_dir=None):

    det_bbox_list = []
    #  "filename":"test_data/test_images/2/0003.jpg",
    image_path = frame_det['filename'] if image_root_dir is None else os.path.join(
        image_root_dir, frame_det['filename']
    )
    assert os.path.isfile(image_path), \
        'Please make sure the images used to' \
        'conduct the detections are still there.'

    dets = frame_det['objects']
    if len(dets) > 0:
        # tmp_img = cv2.imread(image_path)
        img_w, img_h = imagesize.get(image_path)
        for obj in dets:
            class_idx = obj['class_id']
            yolo_bbox_rect = obj['relative_coordinates']
            score = obj['confidence']
            bbox_rect = [
                yolo_bbox_rect['center_x'],
                yolo_bbox_rect['center_y'],
                yolo_bbox_rect['width'],
                yolo_bbox_rect['height'],
            ]
            x1, y1, x2, y2 = yolo_bbox_rect_to_smrc_rect(bbox_rect, img_h, img_w)
            det_bbox_list.append([class_idx, x1, y1, x2, y2, score])
    frame_det_dict = {
        "image_path": frame_det['filename'],   # image_path, keep it as relative path
        "det_bbox_list": det_bbox_list,
    }
    return frame_det_dict


def json_yolov4_to_yolov3(json_yolov4_dir, json_yolov3_dir, image_root_dir=None):
    """
    The image_root_dir can be used to obtain the complete image path
    when the image path in json_yolov4_dir is relative path, i.e.,
    img_path = os.path.join(image_root_dir, frame_det['filename']).

    :param json_yolov4_dir:
    :param json_yolov3_dir:
    :param image_root_dir:
    :return:
    """

    generate_dir_if_not_exist(json_yolov3_dir)
    json_file_list = get_file_list_in_directory(
        json_yolov4_dir, ext_str='.json', only_local_name=True
    )
    for k, json_file in enumerate(json_file_list):
        output_json_file = os.path.join(json_yolov3_dir, json_file)
        input_json_file = os.path.join(json_yolov4_dir, json_file)

        det_list_v3 = []
        with open(input_json_file, 'rb') as input_json:
            det_list_v4 = json.load(input_json)

        for frame_det in det_list_v4:
            # skip the empty detection
            if len(frame_det['objects']) == 0:
                continue

            frame_det_dict = parse_yolov4_frame_det(frame_det, image_root_dir=image_root_dir)
            image_path = frame_det_dict["image_path"]
            for obj in frame_det_dict["det_bbox_list"]:
                class_idx, x1, y1, x2, y2, score = obj
                det_list_v3.append(
                    {"image_path": image_path, "category_id": class_idx, "bbox": [x1, y1, x2, y2], "score": score}
                )

        with open(output_json_file, 'w') as fp:
            fp.write(
                '[\n' +
                ',\n'.join(json.dumps(one_det) for one_det in det_list_v3) +
                '\n]')
        # with open(output_json_file, 'w') as list_v3:
        #     # json.dump dumps all its content in one line.
        #     # indent=2 is to record each dictionary entry on a new line
        #     json.dump(det_list_v3, list_v3, sort_keys=True, indent=2, separators=(',', ':'))
        print(f'Processing {input_json_file} to {output_json_file} done [{k+1}/{len(json_file_list)}]... ')


def load_yolov4_json_detection_to_dict(
        json_detection_file, detection_dict=None,
        score_thd=None, nms_thd=None,
        short_image_path=True, image_root_dir=None
):
    """Load the YOLO detections that are saved in json format to object_detection dict.
    The result can be directly used for object tracking, we do not remove empty detections.
    :param image_root_dir:
    :param json_detection_file: one file include the detections of one video
    :param detection_dict: if not none, continue to add the object_detection to the dict, this
        is useful for ensemble of multiple detections
    :param short_image_path: if true, only save the last two levels of the image path,
        i.e., 3440/0000.jpg
    :param score_thd: remove the low score object_detection, score < score_thd
    :param nms_thd: non maximum suppression threshold
    :return:
        object_detection dict with the format of
            [class_idx, xmin, ymin, xmax, ymax, score]
        A lot of public codes use the format of [class_idx, score, xmin, ymin, xmax, ymax],
        If we do need the public format, just conduct transformation.
    """

    with open(json_detection_file) as json_file:
        json_detection_data = json.load(json_file)

    if detection_dict is None:
        detection_dict = {}

    for frame_det in json_detection_data:  # detection_idx,
        frame_det_dict = parse_yolov4_frame_det(frame_det, image_root_dir=image_root_dir)
        image_path = frame_det_dict["image_path"]
        print(f'{frame_det_dict["image_path"]}')
        if short_image_path:
            image_path = image_path_last_two_level(image_path)

        detection_dict[image_path] = frame_det_dict["det_bbox_list"]

    if score_thd is not None:
        filter_out_low_score_detection_dict(detection_dict, score_thd)

    if nms_thd is not None:
        detection_dict = nms_detection_dict(detection_dict, nms_thd)
    return detection_dict


def yolov4_json_det_to_label(
        json_detection_file, result_label_dir, score_thd=None, nms_thd=None,
        image_root_dir=None
):
    # key is the image path, value is the bbox_list with scores
    detection_dict = load_yolov4_json_detection_to_dict(
        json_detection_file=json_detection_file,
        score_thd=score_thd, nms_thd=nms_thd,
        short_image_path=True, image_root_dir=image_root_dir
    )

    generate_dir_if_not_exist(result_label_dir)
    for image_path in detection_dict:
        ann_path = os.path.join(
            result_label_dir,
            replace_ext_str(old_path=image_path, new_ext_str='.txt')
        )

        bbox_list = det_list_to_bbox_list(detection_dict[image_path])
        if len(bbox_list) > 0:
            generate_dir_if_not_exist(os.path.dirname(ann_path))
            save_bbox_to_file(ann_path, bbox_list)


# ========================================================
# for both yolov3 and yolov4
# ========================================================

def load_yolo_json_det(json_detection_file, det_format, image_root_dir=None,
                       detection_dict=None, score_thd=None, nms_thd=None,
                       short_image_path=True
                       ):
    """
    Load the yolo json detection, if yolov3 format (x1, y1, x2, y2) then we can directly
    extract the box coordinates; if yolov4 format (the image path is saved, but may be
    relative path, and the coordinates are normalized (x1, y1, x2, y2), so we need to use
    the parent image dir (image_root_dir) to get the full image path and get the un-normalized
    bbox coordinates. The final image path will be os.path.join(image_root_dir, image_path), where image_path
    are directly extracted from yolov4 json file.
    Note that if the image path in yolov4 json file is already abs path, then
    just leave the 'image_root_dir' unfilled.

    :param nms_thd:
    :param score_thd:
    :param detection_dict:
    :param json_detection_file:
    :param det_format:
    :param image_root_dir:
    :return:
    """
    assert det_format in ['yolov4', 'yolov3']
    if det_format == 'yolov3':
        detection_dict = load_yolov3_json_detection_to_dict(
            json_detection_file,
            detection_dict=detection_dict,
            score_thd=score_thd,
            nms_thd=nms_thd,
            short_image_path=short_image_path  # True
        )
    elif det_format == 'yolov4':
        if image_root_dir is not None: assert os.path.isfile(image_root_dir)
        detection_dict = load_yolov4_json_detection_to_dict(
            json_detection_file,
            detection_dict=detection_dict,
            score_thd=score_thd,
            nms_thd=nms_thd,
            short_image_path=short_image_path,
            image_root_dir=image_root_dir  #
        )
    else:
        raise NotImplementedError
    return detection_dict


