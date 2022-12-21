import numpy as np
from tqdm import tqdm
import os

from .data_set import load_image_list
from .bbox_metrics import IouSimMat
from .match import HungarianMatch
from .bbox import load_bbox_from_file, compute_iou
from .file_path import get_image_or_annotation_path


# from .image_video import get_image_file_list_in_directory


def estimate_F1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def estimate_precision_recall(TP, FP, FN):
    precision = estimate_precision(TP, FP)
    recall = estimate_recall(TP, FN)
    return precision, recall


def estimate_accu(TP, FP, TN, FN):
    # https://en.wikipedia.org/wiki/Precision_and_recall
    return (TP + TN)/ (TP + FP + TN + FN)


def estimate_precision(TP, FP):
    # https://en.wikipedia.org/wiki/Precision_and_recall
    return TP / (TP + FP)


def estimate_recall(TP, FN):
    # https://en.wikipedia.org/wiki/Precision_and_recall
    return TP / (TP + FN)


def eval_single_image_slow(preds, ground_truth, iou_thd=0.5):
    """
    Compare the prediction (a list of binary mask) and ground_truth (a list of binary mask) and
    estimate the number of tp, fp and fn accordingly.
    We also save the matched iou for reference.
    :param iou_thd:
    :param preds:
    :param ground_truth:
    :return:
    """
    tp, fp, fn, IoUs = 0, 0, 0, []
    no_need_further_check, metric = _early_check(preds, ground_truth)
    if no_need_further_check:
        return metric
    else:
        pred_not_used = [True] * len(preds)
        for gt in ground_truth:
            max_iou = 0
            correct_pred_idx = None
            for i, pred in enumerate(preds):
                iou = compute_iou(pred[1:5], gt[1:5])
                if iou > max_iou:
                    max_iou = iou
                    correct_pred_idx = i

            if max_iou >= iou_thd:
                tp += 1
                pred_not_used[correct_pred_idx] = False
                IoUs.append(max_iou)
            else:
                fn += 1
        fp += np.sum(pred_not_used)
        # for i in range(len(pred_binary_mask_list)):
        #     if pred_not_used[i] is True:
        #         fp += 1
    return [tp, fp, fn, IoUs]


def eval_single_image(preds, gts, iou_thd=0.5):
    no_need_further_check, metric = _early_check(preds, gts)
    if no_need_further_check:
        return metric
    else:
        row_inds, col_inds, iou_mat = _match(preds, gts, iou_thd)

        # row_inds, col_inds = HungarianMatch(1 - iou_mat)  # similarity -> dist
        tp = len(row_inds)
        fn = len(gts) - tp
        fp = len(preds) - tp

        # examples of iou_mat[row_inds, col_inds]: array([1, 2, 2])
        # print(f'row_inds = {row_inds}, col_inds = {col_inds}')
        IoUs = list(iou_mat[row_inds, col_inds]) if tp > 0 else []
        return [tp, fp, fn, IoUs]


def classify_single_img_dets(preds, gts, iou_thd=0.5):
    """
    Classify the predictions to tp, fp for preds, tp and fn for bbox in gts.
    :param preds: 1, tp; 0, fp
    :param gts: 1, tp; 0, fn
    :param iou_thd:
    :return:
    """
    pred_labels = np.zeros(len(preds), dtype=int)
    gt_labels = np.zeros(len(gts), dtype=int)

    # The following lines do nothing.
    # if len(preds) == 0 or len(gts) == 0:
    #     if len(preds) == 0:
    #         gt_labels = np.zeros(len(gts))  #
    #     else:
    #         fp = len(preds)

    if len(preds) > 0 and len(gts) > 0:
        row_inds, col_inds, _ = _match(preds, gts, iou_thd)

        if len(row_inds) > 0: pred_labels[row_inds] = 1
        if len(col_inds) > 0: gt_labels[col_inds] = 1
    return pred_labels, gt_labels


def _match(preds, gts, iou_thd):
    iou_mat = IouSimMat(preds, gts)
    # row_inds, col_inds are 1d arrays. BidirectionalMatch HungarianMatch
    row_inds, col_inds = HungarianMatch(1 - iou_mat, max_distance=1 - iou_thd)
    return row_inds, col_inds, iou_mat


def _early_check(dets, gtruths):
    tp, fp, fn, IoUs = 0, 0, 0, []
    if len(dets) == 0 or len(gtruths) == 0:
        if len(dets) == 0:
            fn = len(gtruths)
        else:  # len(gtruth_lanes) == 0
            fp = len(dets)
        return True, [tp, fp, fn, IoUs]
    else:
        return False, [tp, fp, fn, IoUs]


def load_label_txt_file_list(image_path_list, image_root_dir, label_root_dir):
    # print(f'| Total {len(image_path_list)} image files. ')
    txt_file_list = [
        get_image_or_annotation_path(x, image_root_dir, label_root_dir, '.txt')
        for x in image_path_list
    ]
    return txt_file_list


def get_det_gt_txt_file_list(
        test_image_path_list,
        image_root_dir, det_root_dir, gt_root_dir):
    assert len(test_image_path_list) > 0
    assert os.path.isdir(det_root_dir) and os.path.isdir(gt_root_dir)

    print(f'| Total {len(test_image_path_list)} image files. ')
    det_file_list = load_label_txt_file_list(
        test_image_path_list, image_root_dir=image_root_dir,
        label_root_dir=det_root_dir
    )
    gtruth_file_list = load_label_txt_file_list(
        test_image_path_list, image_root_dir=image_root_dir,
        label_root_dir=gt_root_dir
    )
    return det_file_list, gtruth_file_list


class Eval:
    def __init__(self):
        self._init_metric()

    def _init_metric(self):
        self.TP_ = 0
        self.FP_ = 0
        self.FN_ = 0
        self.IoUs_ = []

    def _update_metric(self, new_metric):
        """
        Update the TP, FP, FN and IoUs for the overall evaluation metrics.
        :param new_metric:
        :return:
        """
        tp, fp, fn, IoUs = new_metric
        self.TP_ += tp
        self.FP_ += fp
        self.FN_ += fn
        self.IoUs_ += IoUs

    def _get_cur_metric(self):
        return [self.TP_, self.FP_, self.FN_, self.IoUs_]

    def _wrap_result(self):
        tp, fp, fn, IoUs = self._get_cur_metric()
        precision, recall = estimate_precision_recall(tp, fp, fn)
        F1_score = estimate_F1_score(precision, recall)
        print(f'| tp = {tp}, fp = {fp}, fn = {fn}, avg_iou = {np.average(IoUs)}')
        print(f'| precision = {precision} \n| recall = {recall}')
        print(f'| F1_score = {F1_score}')
        return precision, recall, F1_score

    def _eval(
            self, det_file_list, gtruth_file_list,
            iou_thd=0.5):
        """
        Evaluate the precision, recall and F1 score.
        :param det_file_list: complete det file list
        :param gtruth_file_list: complete gtruth file list
        :param iou_thd:
        :return:
        """

        history = []
        self._init_metric()
        assert len(gtruth_file_list) == len(det_file_list)
        num_image = len(gtruth_file_list)
        with tqdm(total=num_image) as pbar:
            for det_file, gtruth_file in \
                    zip(det_file_list, gtruth_file_list):
                preds = load_bbox_from_file(det_file)
                gts = load_bbox_from_file(gtruth_file)
                singe_image_metric = eval_single_image(preds=preds, gts=gts, iou_thd=iou_thd)
                self._update_metric(singe_image_metric)
                history.append(
                    [det_file, gtruth_file] + singe_image_metric
                )
                pbar.set_description(f'| tp = {self.TP_}, fp = {self.FP_}, fn = {self.FN_}')
                # f' avg_iou = {np.average(self.IoUs_)}'
                pbar.update(1)
        precision, recall, f1_score = self._wrap_result()

        return history, precision, recall, f1_score

    def eval_img_list(self, test_image_path_list, image_root_dir,
                      det_root_dir, gt_root_dir, iou_thd=0.5):
        det_file_list, gtruth_file_list = get_det_gt_txt_file_list(
            test_image_path_list=test_image_path_list,
            image_root_dir=image_root_dir,
            det_root_dir=det_root_dir,
            gt_root_dir=gt_root_dir)

        return self._eval(
            det_file_list=det_file_list, gtruth_file_list=gtruth_file_list,
            iou_thd=iou_thd
        )

    def eval_img_det_gt_data_root_dir(
            self, image_root_dir, det_root_dir, gt_root_dir, dir_list=None):
        print(f'| Total {len(dir_list)} directories. ')
        test_image_path_list = load_image_list(image_root_dir, dir_list=dir_list)

        return self.eval_img_list(
            test_image_path_list=test_image_path_list,
            image_root_dir=image_root_dir,
            det_root_dir=det_root_dir,
            gt_root_dir=gt_root_dir
        )
