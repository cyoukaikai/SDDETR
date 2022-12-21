import numpy as np
import os
from PIL import ImageGrab


def get_monitor_resolution():
    """Return tuple (width, height), e.g., (1272, 796)
    :return:
    """
    img = ImageGrab.grab()
    return img.size


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)


# get geometric information or relationship
def point_in_rectangle(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom


# get the 8 bboxes of the anchors around the bbox and put them into a dictionary
def get_anchors_rectangles(xmin, ymin, xmax, ymax, line_thickness):
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    bbox_anchor_thickness = line_thickness * 2
    L_ = [xmin - bbox_anchor_thickness, xmin + bbox_anchor_thickness]
    M_ = [mid_x - bbox_anchor_thickness, mid_x + bbox_anchor_thickness]
    R_ = [xmax - bbox_anchor_thickness, xmax + bbox_anchor_thickness]
    _T = [ymin - bbox_anchor_thickness, ymin + bbox_anchor_thickness]
    _M = [mid_y - bbox_anchor_thickness, mid_y + bbox_anchor_thickness]
    _B = [ymax - bbox_anchor_thickness, ymax + bbox_anchor_thickness]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def get_anchors_lane(x1, y1, x2, y2, line_thickness):
    xmin, ymin, xmax, ymax = get_min_rect(x1, y1, x2, y2)
    anchor_list = get_anchors_rectangles(xmin, ymin, xmax, ymax, line_thickness)
    # kept_anchor = ['LT', 'RB']
    # return {key: value for key, value in anchor_list.items() if key in kept_anchor}
    kept_anchor = estimate_line_segment_anchors(x1, y1, x2, y2)
    return {key: anchor_list[key] for key in kept_anchor}


def get_min_rect(x1, y1, x2, y2):
    """
    Return the coordinates of the top left, bottom right points given two points
    (x1, y1), (x2, y2).
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return xmin, ymin, xmax, ymax


def estimate_line_segment_anchors(x1, y1, x2, y2):
    xmin, ymin, xmax, ymax = get_min_rect(x1, y1, x2, y2)

    def estimate_anchor_key(x1_, y1_):
        p_anchor = ''
        if x1_ == xmin: p_anchor += 'L'
        else: p_anchor += 'R'

        if y1_ == ymin: p_anchor += 'T'
        else: p_anchor += 'B'

        return p_anchor

    # kept_anchor = ['LT', 'RB']
    kept_anchor = [estimate_anchor_key(x1, y1), estimate_anchor_key(x2, y2)]
    return kept_anchor

###################################################
# used to save video for object_tracking result
##################################################


def get_annotation_path(img_path, image_dir, label_dir, ann_ext_str):
    new_path = img_path.replace(image_dir, label_dir, 1)
    _, img_ext = os.path.splitext(new_path)
    annotation_path = new_path.replace(img_ext, ann_ext_str, 1)
    # print(annotation_path) #output/0000.txt
    return annotation_path


def split_image_path(image_path):
    file_names = image_path.split(os.path.sep)
    image_name = file_names[-1]
    directory_name = file_names[-2]

    pos = image_path.find(directory_name)
    image_dir = image_path[:pos - 1]
    return image_dir, directory_name, image_name


def estimate_fps_based_on_duration(num_frame, duration):
    fps = np.ceil(num_frame / duration)
    if fps == 0: fps = 1
    print(f'{num_frame} images, np.ceil({num_frame}/15.0) = '
          f'{np.ceil(num_frame /duration)}, fps = {fps} ')
    return fps
