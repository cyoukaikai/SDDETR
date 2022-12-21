############################################
# >>> from sklearn.metrics.pairwise import pairwise_distances
# >>> import numpy as np
# >>> a = np.array([[1,2],[ 3, 4]])
# >>> pdist = pairwise_distances(a, metric="cosine")
# >>> print(pdist)
# [[0.         0.01613009]
#  [0.01613009 0.        ]]
####################################
import numpy as np
import cv2
import random
import sys


from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
# from .test_metric import array_equal, array_close
from .kalman_filter import KalmanFilter
from .bbox import get_bbox_rect, get_bbox_class, bbox_to_xyah, compute_iou
from .annotate.ann_utils import point_in_rectangle


BIG_VALUE = 1e+5
INF_VALUE = float('inf')


def estimate_same_value_mask(vec_a, vec_b):
    """Return a len(vec_a) by len(vec_b) matrix, wherein True if two values are same, false otherwise.
    a = np.array([1, 2, 3])
    b = np.array([1, 5])
    mask = estimate_same_value_mask(a, b)
    print(mask)
            [[ True False]
         [False False]
         [False False]]
    :param vec_a:
    :param vec_b:
    :return:
    """
    # in case the vec_a and vec_b are list
    if isinstance(vec_a, list) or isinstance(vec_b, list):
        vec_a, vec_b = np.array(vec_a), np.array(vec_b)

    m1, m2 = vec_a.shape[0], vec_b.shape[0]  # get the numbers of elements
    mat_ones1, mat_ones2 = np.ones((m2, m1)), np.ones((m1, m2))

    def broadcast_mat_fast(a_vec, b_vec):
        # return a m1 by m2 matrix
        a_mat = np.transpose(a_vec * mat_ones1)
        b_mat = b_vec * mat_ones2
        return a_mat, b_mat

    # *() unpack a tuple as arguments
    mat_a, mat_b = broadcast_mat_fast(vec_a, vec_b)
    mask = mat_a == mat_b

    return mask


def batch_iou_vec(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    assert a.shape == b.shape, \
        f'Given pairs of bbox should have the same shape, but' \
        f' ({a.shape}, {b.shape})..'
    # ' initial' The minimum value of an output element.
    # Must be present to allow computation on empty slice.
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0, initial=-1)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0, initial=-1)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0, initial=-1)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0, initial=-1)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def compute_iou_mat(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
        # np.maximum() is faster than np.max when compare two arrays
        from https://docs.scipy.org/doc/numpy/reference
        /generated/numpy.amax.html#numpy.amax
        "Don’t use amax for element-wise comparison of 2 arrays; when a.shape[0] is 2,
        maximum(a[0], a[1]) is faster than amax(a, axis=0)."

        def broadcast_mat(a_vec, b_vec):
            a_mat = np.transpose(a_vec * mat_ones1)
            b_mat = b_vec * mat_ones2
            return np.array([a_mat, b_mat])
        x1_slow = broadcast_mat(a[:, 0], b[:, 0]).max(axis=0)
        y1_slow = broadcast_mat(a[:, 1], b[:, 1]).max(axis=0)
        x2_slow = broadcast_mat(a[:, 2], b[:, 2]).min(axis=0)
        y2_slow = broadcast_mat(a[:, 3], b[:, 3]).min(axis=0)
        assert np.array_equal(
            np.array([x1_slow, y1_slow, x2_slow, y2_slow]),
            np.array([x1, y1, x2, y2])
        )

        The difference between using np.max and mp.maximum is tiny.
        Basically, np.maximum is faster than np.max, but
        there is almost no difference when num_bbox < 100.
    And it is only 0.09 sec faster when num_bbox = 2800.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    m1, m2 = a.shape[0], b.shape[0]  # get the numbers of bbox
    mat_ones1, mat_ones2 = np.ones((m2, m1)), np.ones((m1, m2))

    def broadcast_mat_fast(a_vec, b_vec):
        a_mat = np.transpose(a_vec * mat_ones1)
        b_mat = b_vec * mat_ones2
        return a_mat, b_mat

    # *() unpack a tuple as arguments
    x1 = np.maximum(*broadcast_mat_fast(a[:, 0], b[:, 0]))
    y1 = np.maximum(*broadcast_mat_fast(a[:, 1], b[:, 1]))
    x2 = np.minimum(*broadcast_mat_fast(a[:, 2], b[:, 2]))
    y2 = np.minimum(*broadcast_mat_fast(a[:, 3], b[:, 3]))

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    AreaAMat, AreaBMat = broadcast_mat_fast(area_a, area_b)
    area_combined = AreaAMat + AreaBMat - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


# iou metric
def IouBboxSim(bbox1, bbox2):
    # compute_iou slightly fast than my implementation
    # smrc.not_used.my_compute_iou (1/2.39 vs 1/2.7)
    # result diff a little, e.g., 0.902 vs 0.904
    return compute_iou(
        bbox1[1:5], bbox2[1:5]
    )


def IouBboxSimVec(bbox_list1, bbox_list2):
    # assert len(bbox_list1) == len(bbox_list2)
    bbox_rects1 = np.array([get_bbox_rect(bbox) for bbox in bbox_list1])
    bbox_rects2 = np.array([get_bbox_rect(bbox) for bbox in bbox_list2])
    return batch_iou_vec(bbox_rects1, bbox_rects2)


def IouSimMat(bbox_list1, bbox_list2):
    bbox_rects1 = np.array([get_bbox_rect(bbox) for bbox in bbox_list1])
    bbox_rects2 = np.array([get_bbox_rect(bbox) for bbox in bbox_list2])
    return compute_iou_mat(bbox_rects1, bbox_rects2)


def IouPairwiseSimMat(bbox_list1):
    # We must handle the diagonal elements, otherwise, the diagonal elements are
    # all 1s (iou with bbox itself is 1)
    dist_mat = IouSimMat(bbox_list1, bbox_list1)
    np.fill_diagonal(dist_mat, 0)
    return dist_mat


# Euclidean distance
def compute_l2_rect(bbox1_rect, bbox2_rect):
    """
    L2 norm, np.linalg.norm([3,4]) = 5.0
    :param bbox1_rect:
    :param bbox2_rect:
    :return:
    """
    return np.linalg.norm(
        np.asarray(bbox1_rect) - np.asarray(bbox2_rect)
    )


def L2BboxDist(bbox_pre, bbox_next, ignore_class_label_diff=True):
    """
    The class label is ignored in default.
    :param bbox_pre:
    :param bbox_next:
    :param ignore_class_label_diff:
    :return:
    """
    if not ignore_class_label_diff and bbox_pre[0] != bbox_next[0]:
        return float("inf")
    else:
        # if ignore the difference of class labels
        dist = compute_l2_rect(bbox_pre[1:5], bbox_next[1:5])
    return dist


def L2FrameBboxDistMat(bbox_list1, bbox_list2, ignore_class_label_diff=True):
    rect_list1 = [get_bbox_rect(bbox) for bbox in bbox_list1]
    rect_list2 = [get_bbox_rect(bbox) for bbox in bbox_list2]
    dist_mat = euclidean_distances(
        np.array(rect_list1), np.array(rect_list2)
    )

    if not ignore_class_label_diff:
        class_list1 = [get_bbox_class(bbox) for bbox in bbox_list1]
        class_list2 = [get_bbox_class(bbox) for bbox in bbox_list2]
        class_dist_mat = np.vstack([np.array(class_list2) != x for x in class_list1])
        dist_mat[class_dist_mat] = INF_VALUE
    return dist_mat


def L2BboxPdistMat(bbox_list, ignore_class_label_diff=True):
    rect_list = [get_bbox_rect(bbox) for bbox in bbox_list]
    dist_mat = pairwise_distances(np.array(rect_list), metric="euclidean")

    if not ignore_class_label_diff:
        # Expected 2D array
        class_list = [[get_bbox_class(bbox)] for bbox in bbox_list]
        label_dist_mat = pairwise_distances(np.array(class_list), metric="euclidean")
        dist_mat[label_dist_mat > 0] = INF_VALUE
    return dist_mat


# def List2HammingPdist(my_list):
# Expected 2D array
#     dist_mat = pairwise_distances(np.array(my_list), metric="euclidean")
#     return dist_mat

######################################
# optical flow distance
######################################


class SamplingMethod:
    Random = 0  # default
    GoodFeaturePoint = 1 
    

# DefaultSampleMethod = SamplingMethod.Random  # default sampling method

NumFeaturePoint = 50
LK_Params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


class ConnectingCriteria:
    PointCount = 0  # default
    IoU = 1

# DefaultConnectingCriteria = ConnectingCriteria.PointCount
# ConnectingCriteria = 'PointCount'
# AvailableConnectingCriteria = ['IoU', 'PointCount']


def get_roi_img_x1y1x2y2(frame, x1, y1, x2, y2):
    # selecting roi (y first, then x)
    roi = frame[y1: y2, x1: x2]
    return roi


def get_bbox_roi_img(frame, bbox):
    """
    Return the image of the region of interest specified by 'bbox'.
    No mater color or gray.
    :param frame: color image or gray image
    :param bbox: format of [class_id, x1, y1, x2, y2]
    :return:
    """
    # x1, y1, x2, y2 = bbox_rect
    # return get_roi_img_x1y1x2y2(frame, x1, y1, x2, y2)

    roi = frame[bbox[2]: bbox[4], bbox[1]: bbox[3]]
    return roi


def sample_feature_points_in_bbox(
        gray_frame, bbox, sampling_method=SamplingMethod.Random,
        num_feature_points=NumFeaturePoint
        ):
    """
    :param gray_frame: 
    :param bbox: 
    :param sampling_method: 
    :param num_feature_points: 
    :return: sampled corners, e.g., 'numpy.ndarray' (7, 1, 2)
        # [
        #   [[301. 229.]]
        #   [[287. 227.]]
        #   ...
        # ] 
    """

    if sampling_method == SamplingMethod.GoodFeaturePoint:
        roi = get_bbox_roi_img(gray_frame, bbox)
        corners = cv2.goodFeaturesToTrack(roi, num_feature_points, 0.01, 10)  # find 50 corfeature points

        # converting to complete image coordinates (new_corners)
        corners[:, 0, 0] = corners[:, 0, 0] + bbox[1]
        corners[:, 0, 1] = corners[:, 0, 1] + bbox[2]

        # we can conduct filtering here for the tracked corner
    elif sampling_method == SamplingMethod.Random:
        corners = np.ndarray(shape=(num_feature_points, 1, 2),
                             dtype=np.float32, order='F')
        x1, y1, x2, y2 = list(map(int, bbox[1:5]))
        xs = random.choices(list(range(x1, x2)), k=num_feature_points)
        ys = random.choices(list(range(y1, y2)), k=num_feature_points)

        # transfer to vector of 2D points
        for i in range(len(xs)):
            corners[i] = np.array([[xs[i], ys[i]]], dtype=np.float32)
    else:
        print('Feature point sampling method not specified ... ')
        sys.exit(0)
    return corners
    
#
# def compute_l2_corner(corner1, corner2):
#     """
#     L2 norm, np.linalg.norm([3,4]) = 5.0
#     :param corner1:  [[301. 229.]]
#     :param corner2:
#     :return:
#     """
#     return np.linalg.norm(
#         np.asarray(corner1) - np.asarray(corner2)
#     )


def track_feature_points(
        oldFrameGray, newFrameGray, old_corners, lk_params=LK_Params
        ):
    """
    old_corners: corners inside a bbox
    """
    new_corners, st, err = cv2.calcOpticalFlowPyrLK(
        oldFrameGray, newFrameGray, old_corners, None, **lk_params
    )

    # transfer float value to int32 or int64
    new_corners = np.int0(new_corners)

    return new_corners


def OptFlowBboxSim(
        color_frame_left, bbox_left,
        color_frame_right, bbox_right,
        sampling_method=SamplingMethod.Random,
        num_feature_points=NumFeaturePoint,
        connecting_criteria=ConnectingCriteria.PointCount,
        bidirectional=False
        ):
    """
    A specific version for "similarity_matrix_given_frames_and_bbox_list"
    """
    # 2 dimension matrix
    similarity_matrix, old_corners_list, new_corners_list = \
        OptFlowFrameSimMat(
            color_frame_left, [bbox_left],
            color_frame_right, [bbox_right],
            sampling_method=sampling_method,
            num_feature_points=num_feature_points,
            connecting_criteria=connecting_criteria,
            bidirectional=bidirectional
        )
    return similarity_matrix[0, 0], old_corners_list, new_corners_list


def OptFlowFrameSimMat(
        color_frame1, bbox_list1,
        color_frame2, bbox_list2,
        sampling_method=SamplingMethod.Random,
        num_feature_points=NumFeaturePoint,
        connecting_criteria=ConnectingCriteria.PointCount,
        bidirectional=False
        ):
    """
    For sparse scene (not many clutter), double directional optical flow
    has no notable effect.
    tests example:
        color_image1 = cv2.imread('0000.jpg')
        color_image2 = cv2.imread('0001.jpg')
        bbox_list1 = smrc.not_used.load_bbox_from_file('0000.txt')
        bbox_list2 = smrc.not_used.load_bbox_from_file('0001.txt')

        np.savetxt('diff-forward-backward',
        similar_matrix_forward - np.transpose(similar_matrix_backward),
        fmt='%.3f')

        0.020 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 -0.060 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 -0.060 0.000 0.000 0.040 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 -0.020 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 -0.080 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.060 0.000 0.000 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.060 0.060 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.080 0.000 0.000 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 -0.060 0.000 0.000 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 -0.020 0.000
        0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
    """
    assert len(bbox_list1) > 0 and len(bbox_list2) > 0

    def computer_forward_similarity_matrix(
            color_frame_left, bbox_list_left,
            color_frame_right, bbox_list_right
    ):
        similarity_matrix = np.zeros((len(bbox_list_left), len(bbox_list_right)),
                                     dtype=np.float32)

        oldFrameGray = cv2.cvtColor(color_frame_left, cv2.COLOR_BGR2GRAY)
        newFrameGray = cv2.cvtColor(color_frame_right, cv2.COLOR_BGR2GRAY)

        # old_corners for object_tracking, and tracked corners from old corners
        old_corners_list = [None] * len(bbox_list_left)
        new_corners_list = [None] * len(bbox_list_left)

        for i, bbox_prev in enumerate(bbox_list_left):
            old_corners_list[i] = sample_feature_points_in_bbox(
                oldFrameGray, bbox_prev,
                sampling_method, num_feature_points
            )
        # this method does not work for tracking good feature points
        # as there may be less than the expected number of points sampled
        # e.g., 7 points are sampled when we want to sample 50 points
        # sometimes, we may have 0 point sampled.
        old_corners_all = np.vstack(old_corners_list)
        tracked_corners_all = track_feature_points(
            oldFrameGray, newFrameGray, old_corners_all
        )

        for i, bbox_prev in enumerate(bbox_list_left):
            str_id, end_id = num_feature_points * i, num_feature_points * (i + 1)

            # extract the tracked_corners
            new_corners_list[i] = tracked_corners_all[str_id:end_id, :, :]

            similarity_list = similarity_tracked_corners_and_bbox_list(
                new_corners_list[i], bbox_list_right,
                connecting_criteria, num_feature_points
            )
            for j, x in enumerate(similarity_list):
                similarity_matrix[i, j] = x
        return similarity_matrix, old_corners_list, new_corners_list

    similar_matrix_forward, corners_list1, corners_list2 = \
        computer_forward_similarity_matrix(
            color_frame_left=color_frame1, bbox_list_left=bbox_list1,
            color_frame_right=color_frame2, bbox_list_right=bbox_list2
        )

    if bidirectional:
        similar_matrix_backward, _, _ = computer_forward_similarity_matrix(
            color_frame_left=color_frame2, bbox_list_left=bbox_list2,
            color_frame_right=color_frame1, bbox_list_right=bbox_list1
        )
        final_similar_matrix = np.add(
            similar_matrix_forward, np.transpose(similar_matrix_backward)
        ) / 2.0
    else:
        final_similar_matrix = similar_matrix_forward

    return final_similar_matrix, corners_list1, corners_list2


def OptFlowFrameDistMat(
        color_frame1, bbox_list1,
        color_frame2, bbox_list2,
        sampling_method=SamplingMethod.Random,
        num_feature_points=NumFeaturePoint,
        connecting_criteria=ConnectingCriteria.PointCount,
        bidirectional=False
    ):
    final_similar_matrix, _, _ = OptFlowFrameSimMat(
        color_frame1=color_frame1, bbox_list1=bbox_list1,
        color_frame2=color_frame2, bbox_list2=bbox_list2,
        sampling_method=sampling_method,
        num_feature_points=num_feature_points,
        connecting_criteria=connecting_criteria,
        bidirectional=bidirectional
    )
    return 1 - final_similar_matrix


def minBoundingRectForPoints(corners):
    # corners, a list of 2 d arrays of the shape of (1,2)
    # e.g.,
    # [
    #   [[301. 229.]]
    #   [[287. 227.]]
    #   ...
    # ]
    Xs = [i[0, 0] for i in corners]
    Ys = [i[0, 1] for i in corners]

    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    return [round(x1), round(y1), round(x2), round(y2)]


# modify this function once new connecting criteria is added.
def similarity_tracked_corners_and_bbox_list(
        tracked_new_corners, bbox_list,
        connecting_criteria=ConnectingCriteria.PointCount,
        num_feature_points=NumFeaturePoint
        ):
    # raise NotImplementedError

    # the similarity must be initialized as 0
    similarity_list = [0] * len(bbox_list)

    if connecting_criteria == ConnectingCriteria.IoU:
        # if less than 2 tracked points.
        if len(tracked_new_corners) < 2:
            return similarity_list

        tracked_bbox_rect = minBoundingRectForPoints(tracked_new_corners)
        for j, bbox in enumerate(bbox_list):
            similarity_list[j] = compute_iou(
                bbox[1:5], tracked_bbox_rect)
    elif connecting_criteria == ConnectingCriteria.PointCount:
        for feature_point in tracked_new_corners:
            # corners, a list of 2 d arrays of the shape of (1,2)
            # print(feature_point) #[[414.09253 245.0596 ]]
            pX, pY = round(feature_point[0, 0]), round(feature_point[0, 1])
            # print(f'px = {pX}, py = {pY}')

            for j, bbox_new in enumerate(bbox_list):
                x1, y1, x2, y2 = bbox_new[1:]
                if point_in_rectangle(pX, pY, x1, y1, x2, y2):
                    similarity_list[j] += 1
                # print(f'j = {j}, bbox_new = {bbox_new}')

        # transfer the point count to similarity so that it is in [0,1]
        similarity_list = [x / num_feature_points for x in similarity_list]

    return similarity_list


############################################
# cosine distance of extracted features
############################################


def FeatureBboxCosineDist(feature_vector1, feature_vector2):
    """
    Cosine distance between extracted features of two bounding boxes.
    :param feature_vector1: features of bbox 1
    :param feature_vector2: features of bbox 2
    :return: 
    """
    cosine_dist_mat = FeatureFrameCosineDistMat(feature_vector1, feature_vector2)
    return np.asscalar(cosine_dist_mat)


def FeatureFrameCosineDistMat(feature_matrix1, feature_matrix2):
    """Compute pair-wise cosine distance between the feature lists
    of two cluster/tracks.
    The code is borrowed from deep_sort/nn_matching.py
    Parameters
    ----------
    feature_matrix1 : array_like
        An NxM matrix of N samples of dimensionality M.
    feature_matrix2 : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """

    feature_matrix1 = np.asarray(feature_matrix1) / np.linalg.norm(feature_matrix1, axis=1, keepdims=True)
    feature_matrix2 = np.asarray(feature_matrix2) / np.linalg.norm(feature_matrix2, axis=1, keepdims=True)
    return 1. - np.dot(feature_matrix1, feature_matrix2.T)


def FeatureCosinePdistMat(feature_matrix):
    """
    Each row in feature_matrix represent a sample.
    :param feature_matrix:
    :return:
    """
    pdist = pairwise_distances(feature_matrix, metric="cosine")
    return pdist


class BBoxKF:
    """A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.

    """

    def __init__(self, mean, covariance):
        self.mean = mean
        self.covariance = covariance

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

    def update(self, kf, detection_bbox_xyah):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection_bbox_xyah : the new observed detection
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection_bbox_xyah)

    def gating_dist_between_kf_and_same_frame_dets(self, kf, detections_bbox_xyah, only_position=False):
        """
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        :param kf:
        :param detections_bbox_xyah: distance between an existing cluster and detections in a (new) frame
        :param only_position:
        :return:
        """
        measurements = np.asarray(detections_bbox_xyah).reshape(-1, 4)
        distance = kf.gating_distance(
            self.mean, self.covariance,
            measurements=measurements,
            only_position=only_position
        )
        return distance

    def mahalanobis_dist(self, kf, detection_bbox_xyah, only_position=False):
        """Return the gating distance between kf and a single detection.

        :param kf:
        :param detection_bbox_xyah:
        :param only_position:
        :return:
        """
        distance = self.gating_dist_between_kf_and_same_frame_dets(
            kf, detection_bbox_xyah, only_position=only_position
        ).item()
        return distance


class BBoxKFDist:
    SkipEmptyDet = False
    """
    SkipEmptyDet: if True, we do not update kf state once no observation comes, then
        the distance of the state of kf with future new detection is very likely to be large, as no state
        update is conducted. So 'True' is a strong condition for object_tracking, i.e.,
         no allowed sharp change in location.
         We should not skip empty detection for kf in the general sense.
    """
    @staticmethod
    def sort_bbox_data(bbox_data):
        """
        :param bbox_data: data format: [ [image_id, bbox], ... ]
        :return:
        """
        bbox_data.sort(key=lambda r: r[0])
        return bbox_data

    @staticmethod
    def get_image_ids_from_bbox_data(bbox_data):
        return [x[0] for x in bbox_data]

    @staticmethod
    def get_bbox_list_from_bbox_data(bbox_data):
        return [x[1] for x in bbox_data]

    def kf_gating_dist_list(self, bbox_data, skip_empty_detection=SkipEmptyDet):
        """
        :param bbox_data: data format: [ [image_id, bbox], ... ]
        :param skip_empty_detection: if True, we do not update kf state once no observation comes, then
        the distance of the state of kf with future new detection is very likely to be large, as no state
        update is conducted. So 'True' is a strong condition for object_tracking, i.e.,
         no allowed sharp change in location.
         In general, we should not skip empty detection for kf in the general sense.
        :return: gating_distance_list of length len(bbox_data) -1
            (as the distance for the first element is 0)
        """
        assert len(bbox_data) > 1, 'There should be at least two bbox for estimating the kf dist.'
        bbox_data = self.sort_bbox_data(bbox_data)
        bbox_list = self.get_bbox_list_from_bbox_data(bbox_data)
        image_id_list = self.get_image_ids_from_bbox_data(bbox_data)

        kf, kf_track = self.init_bbox_kf_with_one_bbox(bbox=bbox_list[0])

        gating_distance_list = []
        # for image_id in range(min(image_id_list)+1, max(image_id_list)+1):
        for i in range(1, len(bbox_list)):
            times_update = image_id_list[i] - image_id_list[i-1]

            kf_track.predict(kf)
            if times_update > 1 and not skip_empty_detection:
                for j in range(1, times_update):
                    kf_track.predict(kf)

            bbox = bbox_list[i]
            detection_bbox_xyah = bbox_to_xyah(bbox, with_class_index=False)

            distance = kf_track.mahalanobis_dist(
                kf=kf, detection_bbox_xyah=detection_bbox_xyah
            )
            # print(f'kf fitting the {ind}th bbox on the cluster, distance = {distance} ...')
            gating_distance_list.append(distance)

            # update the kalman filter only if new observations arrive
            kf_track.update(kf, detection_bbox_xyah)

        return gating_distance_list

    @staticmethod
    def init_bbox_kf_with_one_bbox(bbox):
        kf = KalmanFilter()
        mean, covariance = kf.initiate(
            bbox_to_xyah(bbox, with_class_index=False)
        )

        # initialize kf with the first bbox in cluster_left
        kf_track = BBoxKF(
            mean=mean, covariance=covariance
        )
        return kf, kf_track

    def bbox_kf_left_right_dist(self, bbox_data_left, bbox_data_right, skip_empty_detection=SkipEmptyDet):

        bbox_data = bbox_data_left + bbox_data_right
        assert len(bbox_data) > 1

        # get the new ids for the right cluster
        sorted_ids = sorted(
            range(len(bbox_data)),
            key=lambda r: bbox_data[r][0]  # sort based on image id
        )

        # check if the first element of the sorted merged cluster is indeed in the left cluster
        assert sorted_ids[0] < len(bbox_data_left)

        # appending a None element for the first element to make the length of the gating_dist_list
        # equal to the length of the bbox_data
        gating_dist_list = [None] + self.kf_gating_dist_list(
            bbox_data, skip_empty_detection
        )

        # extract the dist for the right cluster
        gating_dist_indices = [sorted_ids.index(k) for k in range(len(bbox_data_left), len(bbox_data))]
        right_cluster_dist_list = [gating_dist_list[k] for k in gating_dist_indices]

        return right_cluster_dist_list

