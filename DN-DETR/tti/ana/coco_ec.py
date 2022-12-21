# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
# from mmdet.datasets.custom import CustomDataset
from mmdet.datasets import CocoDataset
# ===================
import smrc.utils
import os
from tti.mm2det._conf import MM2DET_RESULT_DIR
# =================
import copy


@DATASETS.register_module()
class CocoECDataset(CocoDataset):  # ErrorCorrecting
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 output_dir=None,  # os.path.join(MM2DET_RESULT_DIR, 'detr')
                 ):

        # evaluate_original
        return self.evaluate_error_corrected(
            results=results,
            metric=metric,
            logger=logger,
            jsonfile_prefix=jsonfile_prefix,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            output_dir=output_dir,  # os.path.join(MM2DET_RESULT_DIR, 'detr')
        )

    def evaluate_error_corrected(self,
                                 results,
                                 metric='bbox',
                                 logger=None,
                                 jsonfile_prefix=None,
                                 classwise=False,
                                 proposal_nums=(100, 300, 1000),
                                 iou_thrs=None,
                                 metric_items=None,
                                 output_dir=None,  # os.path.join(MM2DET_RESULT_DIR, 'detr')
                                 ):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            # iou_thrs = np.linspace(
            #     .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            # # ----------------------
            # iou_thrs = np.linspace(
            #     .45, 0.9, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            # iou_thrs = np.linspace(
            #     .05, 0.95, int(np.round((0.95 - .05) / .05)) + 1, endpoint=True)
            iou_thrs = np.linspace(
                .05, 0.1, int(np.round((0.1 - .05) / .05)) + 1, endpoint=False)
            # # ----------------------
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # ==============================
        if jsonfile_prefix is None and output_dir is not None:
            jsonfile_prefix = os.path.join(output_dir + '_results')
        # ==========================

        tmp_dir = None
        result_files = dict()
        bbox_file = f'{jsonfile_prefix}.bbox.json'
        result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
        result_files['proposal'] = f'{jsonfile_prefix}.bbox.json'
        # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        if not os.path.isfile(bbox_file):
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)  # load annotation

            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:

                cocoEval_matched_pred_file = None
                if output_dir is not None:
                    cocoEval_matched_pred_file = os.path.join(output_dir + '_cocoEval_matched_pred.pkl')

                if cocoEval_matched_pred_file is not None and os.path.isfile(cocoEval_matched_pred_file):
                    print(f'File {cocoEval_matched_pred_file} found. ')
                    cocoEval = smrc.utils.load_pkl_file(cocoEval_matched_pred_file)
                else:
                    cocoEval.evaluate()
                    if cocoEval_matched_pred_file is not None:
                        smrc.utils.generate_pkl_file(
                            pkl_file_name=cocoEval_matched_pred_file,
                            data=cocoEval
                        )
                cocoEval_back = copy.deepcopy(cocoEval)

                result_display = []
                for iou in [0.0, 0.5]:
                    for confidence in np.arange(0, 0.95, 0.05):
                        for ratio in np.arange(0, 0.5, 0.05):
                            print(f'================================================= \n'
                                  # f'{result_files["bbox"]} \n'
                                  f'correction ratio for all matched predictions = {ratio}, '
                                  f'tp_iou_thd = {iou} confidence = {confidence}\n'
                                  f'========================================================= ')

                            if ratio > 0:
                                cocoEval, avg_error_corrected = self.correct_tp_error(copy.deepcopy(cocoEval_back),
                                                                 ratio=ratio, iou_thd=iou,
                                                                 score_thd=confidence,
                                                                 output_dir=output_dir)
                            else:
                                cocoEval, avg_error_corrected = copy.deepcopy(cocoEval_back), [0, 0]
                            # change the iou_thrs to normal
                            iou_thrs = np.linspace(
                                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
                            cocoEval.params.iouThrs = iou_thrs
                            print(f'| Evaluating ...')
                            cocoEval.evaluate()
                            # # ===================================================
                            # self.estimate_boundary_error(cocoEval, output_dir)
                            # # ===================================================
                            print(f'| Acumulate ...')
                            cocoEval.accumulate()

                            # Save coco summarize print information to logger
                            redirect_string = io.StringIO()
                            with contextlib.redirect_stdout(redirect_string):
                                cocoEval.summarize()
                            print_log('\n' + redirect_string.getvalue(), logger=logger)

                            if classwise:  # Compute per-category AP
                                # Compute per-category AP
                                # from https://github.com/facebookresearch/detectron2/
                                precisions = cocoEval.eval['precision']
                                # precision: (iou, recall, cls, area range, max dets)
                                assert len(self.cat_ids) == precisions.shape[2]

                                results_per_category = []
                                for idx, catId in enumerate(self.cat_ids):
                                    # area range index 0: all area ranges
                                    # max dets index -1: typically 100 per image
                                    nm = self.coco.loadCats(catId)[0]
                                    precision = precisions[:, :, idx, 0, -1]
                                    precision = precision[precision > -1]
                                    if precision.size:
                                        ap = np.mean(precision)
                                    else:
                                        ap = float('nan')
                                    results_per_category.append(
                                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                                num_columns = min(6, len(results_per_category) * 2)
                                results_flatten = list(
                                    itertools.chain(*results_per_category))
                                headers = ['category', 'AP'] * (num_columns // 2)
                                results_2d = itertools.zip_longest(*[
                                    results_flatten[i::num_columns]
                                    for i in range(num_columns)
                                ])
                                table_data = [headers]
                                table_data += [result for result in results_2d]
                                table = AsciiTable(table_data)
                                print_log('\n' + table.table, logger=logger)

                            if metric_items is None:
                                metric_items = [
                                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                                ]

                            for metric_item in metric_items:
                                key = f'{metric}_{metric_item}'
                                val = float(
                                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                                )
                                eval_results[key] = val
                            ap = cocoEval.stats[:6]
                            eval_results[f'{metric}_mAP_copypaste'] = (
                                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                                f'{ap[4]:.3f} {ap[5]:.3f}')
                            print(eval_results[f'{metric}_mAP_copypaste'])

                            result_display.append([
                                ratio, iou, *eval_results[f'{metric}_mAP_copypaste'].split(' '), *avg_error_corrected
                            ])
                            # {ratio}, tp_iou_thd = {iou}

                if output_dir is not None:
                    result_display_file = os.path.join(output_dir + '_error_correct_eval_results.txt')
                    smrc.utils.save_multi_dimension_list_to_file(result_display_file,
                                                                 list_to_save=result_display, delimiter=' ')

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def correct_tp_error(self, cocoEval: COCOeval, ratio=0.1, iou_thd: float = 0.0,
                         score_thd=0,
                         output_dir: str = None):
        """
         cocoEval.evalImgs = [ {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }, ..., ]
        """

        # only all scales, based on confidence
        p = cocoEval.params

        errors_l_r_t_b = []  # image id, det id, det box, matched gt id,
        count = 0
        for k, eval_img in enumerate(cocoEval.evalImgs):
            if eval_img is None or eval_img['aRng'] != p.areaRng[0] or eval_img['maxDet'] != 1000:
                continue
            else:
                for i in range(len(eval_img['dtIds'])):
                    det_id = eval_img['dtIds'][i]
                    det_box = cocoEval.cocoDt.anns[det_id]['bbox']
                    # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2
                    det_x1, det_x2, det_y1, det_y2 = [det_box[0], det_box[0] + det_box[2],
                                                      det_box[1], det_box[1] + det_box[3]]
                    # # =============== iou related section
                    # # skip ignored detection
                    # if eval_img['dtIgnore'][0][i]:
                    #     continue

                    matched_gt_id = eval_img['dtMatches'][0][i]  # iou thd = 0.05
                    # ===============
                    imgId, catId = eval_img['image_id'], eval_img['category_id'],
                    if matched_gt_id != 0:  # matched with gt
                        gt_box = cocoEval.cocoGt.anns[matched_gt_id]['bbox']
                        gt_x1, gt_x2, gt_y1, gt_y2 = [gt_box[0], gt_box[0] + gt_box[2],
                                                      gt_box[1], gt_box[1] + gt_box[3]]
                        error_abs = list(np.abs(np.array([det_x1, det_x2, det_y1, det_y2]) -
                                            np.array([gt_x1, gt_x2, gt_y1, gt_y2])))
                        error = np.array([det_x1, det_x2, det_y1, det_y2]) - np.array([gt_x1, gt_x2, gt_y1, gt_y2])

                        # # The saved iou has different order
                        computed_ious = cocoEval.ious[(imgId, catId)]
                        g_id = eval_img['gtIds'].index(matched_gt_id)
                        iou = computed_ious[i, g_id]
                        if iou < iou_thd:
                            continue

                        if eval_img['dtScores'][i] < score_thd:
                            continue
                        # iou_estimated = smrc.utils.bbox.compute_iou(
                        #     [det_x1, det_y1, det_x2, det_y2],
                        #     [gt_x1, gt_y1, gt_x2, gt_y2]
                        # )
                        #
                        # modify the prediction
                        # det_box = cocoEval.cocoDt.anns[det_id]['bbox']
                        # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2

                        count += 1
                        det_x1_new, det_x2_new, det_y1_new, det_y2_new = \
                            np.array([det_x1, det_x2, det_y1, det_y2]) - error * ratio

                        cocoEval.cocoDt.anns[det_id]['bbox'] = [det_x1_new, det_y1_new,
                                                                det_x2_new - det_x1_new,
                                                                det_y2_new - det_y1_new
                                                                ]
                        single_det_error_info = [
                            eval_img['dtIds'][i],
                            eval_img['dtScores'][i],
                            np.mean(np.array(error_abs)), np.max(np.array(error_abs)),
                            *error_abs,
                            matched_gt_id,
                            iou,
                            eval_img['dtIgnore'][0][i],  # ignore flag

                        ]
                        errors_l_r_t_b.append(single_det_error_info)
                    # else:
                    #     print('not matched')
                    #     # gt_x1, gt_x2, gt_y1, gt_y2 = -1, -1, -1, -1
                    #     # error = [-100, -100, -100, -100]
                    #     # iou = 0
                    #     # iou_estimated = 0

                    # single_det_error_info = [
                    #     eval_img['dtIds'][i],
                    #     eval_img['dtScores'][i],
                    #     matched_gt_id,
                    #     imgId, catId,
                    #     # eval_img['aRng'],
                    #     eval_img['maxDet'],
                    #     iou,
                    #     iou_estimated,
                    #     eval_img['dtIgnore'][0][i],  # ignore flag
                    #     *error,
                    #     np.mean(np.array(error)), np.max(np.array(error)),
                    #     det_x1, det_x2, det_y1, det_y2,
                    #     gt_x1, gt_x2, gt_y1, gt_y2,
                    # ]
                    # errors_l_r_t_b.append(single_det_error_info)

        # # sorted errors_l_r_t_b based on scores
        # n = len(errors_l_r_t_b)
        # sorted_errors_l_r_t_b = sorted(errors_l_r_t_b, key=lambda x: x[1], reverse=True)
        # matched_det_n = [x[2] != 0 for x in sorted_errors_l_r_t_b]
        # tp_ratio = list(np.cumsum(np.array(matched_det_n)) / np.array(range(1, len(matched_det_n) + 1)))
        # sorted_errors_l_r_t_b = [e + [r] for e, r in zip(sorted_errors_l_r_t_b, tp_ratio)]
        # print(f'===================================== \n '
        #       f'Total {len(errors_l_r_t_b)} detections. ')
        if output_dir is not None:
        #     smrc.utils.generate_pkl_file(
        #         pkl_file_name=os.path.join(output_dir + '_cocoEval.pkl'),
        #         data=cocoEval
        #     )
        #     smrc.utils.generate_pkl_file(
        #         pkl_file_name=os.path.join(output_dir + '_errors_l_r_t_b.pkl'),
        #         data=sorted_errors_l_r_t_b
        #     )
            smrc.utils.save_multi_dimension_list_to_file(
                filename=os.path.join(output_dir + f'_errors_tp_iou{iou_thd}_l_r_t_b.txt'),
                list_to_save=errors_l_r_t_b
            )
        average_error = np.mean(np.array(errors_l_r_t_b), axis=0)[1:4]
        return_error = average_error[1:] * ratio

        average_error = [smrc.utils.float_to_str(x, num_digits=2) for x in list(average_error)]
        # {ap[1]: .3f}
        print(f'count = {count} modified, mean, max error {average_error}')
        return cocoEval, list(return_error)

    def evaluate_original(self,
                          results,
                          metric='bbox',
                          logger=None,
                          jsonfile_prefix=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None,
                          output_dir=None,  # os.path.join(MM2DET_RESULT_DIR, 'detr')
                          ):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            # # ----------------------
            # iou_thrs = np.linspace(
            #     .5, 0.6, int(np.round((0.6 - .5) / .05)) + 1, endpoint=True)
            # # ----------------------
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # ==============================
        if jsonfile_prefix is None and output_dir is not None:
            jsonfile_prefix = os.path.join(output_dir + '_results')

            # ==========================
        tmp_dir = None
        result_files = dict()
        bbox_file = f'{jsonfile_prefix}.bbox.json'
        result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
        result_files['proposal'] = f'{jsonfile_prefix}.bbox.json'
        # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        if not os.path.isfile(bbox_file):
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)  # load annotation

            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                # ===================================================
                self.estimate_boundary_error(cocoEval, output_dir)
                # ===================================================
                cocoEval.accumulate()

                # Save coco summarize print information to logger
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                print_log('\n' + redirect_string.getvalue(), logger=logger)

                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def evaluate_analyze(self,
                         results,
                         metric='bbox',
                         logger=None,
                         jsonfile_prefix=None,
                         classwise=False,
                         proposal_nums=(100, 300, 1000),
                         iou_thrs=None,
                         metric_items=None,
                         output_dir=None,  # os.path.join(MM2DET_RESULT_DIR, 'detr')
                         ):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            # # ----------------------
            # iou_thrs = np.linspace(
            #     .5, 0.6, int(np.round((0.6 - .5) / .05)) + 1, endpoint=True)
            # # ----------------------
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        # ==============================
        if jsonfile_prefix is None and output_dir is not None:
            jsonfile_prefix = os.path.join(output_dir + '_results')

            # ==========================
        tmp_dir = None
        result_files = dict()
        bbox_file = f'{jsonfile_prefix}.bbox.json'
        result_files['bbox'] = f'{jsonfile_prefix}.bbox.json'
        result_files['proposal'] = f'{jsonfile_prefix}.bbox.json'
        # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        if not os.path.isfile(bbox_file):
            result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        # =================================
        conf_thds = list(np.arange(0., 1.0, 0.05))
        for conf_value in conf_thds:
            # =========================
            print(f'================================================= \n'
                  f'{result_files["bbox"]} \n'
                  f'conf_value = {conf_value} \n'
                  f'========================================================= ')

            eval_results = OrderedDict()
            cocoGt = self.coco
            for metric in metrics:
                msg = f'Evaluating {metric}...'
                if logger is None:
                    msg = '\n' + msg
                print_log(msg, logger=logger)

                if metric == 'proposal_fast':
                    ar = self.fast_eval_recall(
                        results, proposal_nums, iou_thrs, logger='silent')
                    log_msg = []
                    for i, num in enumerate(proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
                        log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                    log_msg = ''.join(log_msg)
                    print_log(log_msg, logger=logger)
                    continue

                iou_type = 'bbox' if metric == 'proposal' else metric
                if metric not in result_files:
                    raise KeyError(f'{metric} is not in results')
                try:
                    predictions = mmcv.load(result_files[metric])
                    if iou_type == 'segm':
                        # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                        # When evaluating mask AP, if the results contain bbox,
                        # cocoapi will use the box area instead of the mask area
                        # for calculating the instance area. Though the overall AP
                        # is not affected, this leads to different
                        # small/medium/large mask AP results.
                        for x in predictions:
                            x.pop('bbox')
                        warnings.simplefilter('once')
                        warnings.warn(
                            'The key "bbox" is deleted for more accurate mask AP '
                            'of small/medium/large instances since v2.12.0. This '
                            'does not change the overall mAP calculation.',
                            UserWarning)

                        # Conduct filtering with respect to the confidence.
                    """ a list of dict
                    {'image_id': 397133, 'bbox': [158.986328125, 259.1803283691406, 13.001617431640625, 16.18701171875],
                     'score': 0.07025142759084702, 'category_id': 50}
                     
                           {'image_id': 507223, 'bbox': [87.47966766357422, 0.07363319396972656, 
                           71.43360137939453, 54.066476821899414], 
                           'score': 0.9821773767471313, 'category_id': 1, 
                           'segmentation': [[87.47966766357422, 0.07363319396972656, 87.47966766357422, 54.14011001586914, 
                           158.91326904296875, 54.14011001586914, 158.91326904296875, 0.07363319396972656]], 
                           'area': 3862.1631532838364, 'id': 23104, 'iscrowd': 0}
                    """
                    num_before_filtering = len(predictions)
                    predictions = [det for det in predictions if det['score'] >= conf_value]
                    print(f'| Before filtering {num_before_filtering} detections, '
                          f'after filtering {len(predictions)} predictions.')
                    cocoDt = cocoGt.loadRes(predictions)  # load annotation

                except IndexError:
                    print_log(
                        'The testing results of the whole dataset is empty.',
                        logger=logger,
                        level=logging.ERROR)
                    break
                cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
                cocoEval.params.catIds = self.cat_ids
                cocoEval.params.imgIds = self.img_ids
                cocoEval.params.maxDets = list(proposal_nums)
                cocoEval.params.iouThrs = iou_thrs
                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
                if metric_items is not None:
                    for metric_item in metric_items:
                        if metric_item not in coco_metric_names:
                            raise KeyError(
                                f'metric item {metric_item} is not supported')

                if metric == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.evaluate()
                    cocoEval.accumulate()

                    # Save coco summarize print information to logger
                    redirect_string = io.StringIO()
                    with contextlib.redirect_stdout(redirect_string):
                        cocoEval.summarize()
                    print_log('\n' + redirect_string.getvalue(), logger=logger)

                    if metric_items is None:
                        metric_items = [
                            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                            'AR_m@1000', 'AR_l@1000'
                        ]

                    for item in metric_items:
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                        eval_results[item] = val
                else:
                    cocoEval.evaluate()
                    # ===================================================
                    self.estimate_boundary_error(cocoEval, output_dir + f'{conf_value}')
                    # ===================================================
                    cocoEval.accumulate()

                    # Save coco summarize print information to logger
                    redirect_string = io.StringIO()
                    with contextlib.redirect_stdout(redirect_string):
                        cocoEval.summarize()
                    print_log('\n' + redirect_string.getvalue(), logger=logger)

                    if classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = cocoEval.eval['precision']
                        # precision: (iou, recall, cls, area range, max dets)
                        assert len(self.cat_ids) == precisions.shape[2]

                        results_per_category = []
                        for idx, catId in enumerate(self.cat_ids):
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = self.coco.loadCats(catId)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            results_per_category.append(
                                (f'{nm["name"]}', f'{float(ap):0.3f}'))

                        num_columns = min(6, len(results_per_category) * 2)
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = ['category', 'AP'] * (num_columns // 2)
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        print_log('\n' + table.table, logger=logger)

                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                        ]

                    for metric_item in metric_items:
                        key = f'{metric}_{metric_item}'
                        val = float(
                            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                        )
                        eval_results[key] = val
                    ap = cocoEval.stats[:6]
                    eval_results[f'{metric}_mAP_copypaste'] = (
                        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                        f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

    def estimate_boundary_error(self, cocoEval: COCOeval, output_dir: str = None):
        """
         cocoEval.evalImgs = [ {
            'image_id':     imgId,
            'category_id':  catId,
            'aRng':         aRng,
            'maxDet':       maxDet,
            'dtIds':        [d['id'] for d in dt],
            'gtIds':        [g['id'] for g in gt],
            'dtMatches':    dtm,
            'gtMatches':    gtm,
            'dtScores':     [d['score'] for d in dt],
            'gtIgnore':     gtIg,
            'dtIgnore':     dtIg,
        }, ..., ]
        """

        # only all scales, based on confidence
        p = cocoEval.params
        # T = len(p.iouThrs)
        # # G = len(gt)
        # # D = len(dt)

        errors_l_r_t_b = []  # image id, det id, det box, matched gt id,
        for k, eval_img in enumerate(cocoEval.evalImgs):
            if eval_img is None or eval_img['aRng'] != p.areaRng[0] or eval_img['maxDet'] != 1000:
                continue
            else:
                for i in range(len(eval_img['dtIds'])):
                    det_id = eval_img['dtIds'][i]
                    det_box = cocoEval.cocoDt.anns[det_id]['bbox']
                    # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2
                    det_x1, det_x2, det_y1, det_y2 = [det_box[0], det_box[0] + det_box[2],
                                                      det_box[1], det_box[1] + det_box[3]]
                    # # =============== iou related section
                    # # skip ignored detection
                    # if eval_img['dtIgnore'][0][i]:
                    #     continue

                    matched_gt_id = eval_img['dtMatches'][0][i]  # iou thd = 0.5
                    # ===============

                    imgId, catId = eval_img['image_id'], eval_img['category_id'],
                    if matched_gt_id != 0:  # matched with gt
                        gt_box = cocoEval.cocoGt.anns[matched_gt_id]['bbox']
                        gt_x1, gt_x2, gt_y1, gt_y2 = [gt_box[0], gt_box[0] + gt_box[2],
                                                      gt_box[1], gt_box[1] + gt_box[3]]
                        error = list(np.abs(np.array([det_x1, det_x2, det_y1, det_y2]) -
                                            np.array([gt_x1, gt_x2, gt_y1, gt_y2])))

                        # The saved iou has different order
                        computed_ious = cocoEval.ious[(imgId, catId)]
                        g_id = eval_img['gtIds'].index(matched_gt_id)
                        iou = computed_ious[i, g_id]

                        iou_estimated = smrc.utils.bbox.compute_iou(
                            [det_x1, det_y1, det_x2, det_y2],
                            [gt_x1, gt_y1, gt_x2, gt_y2]
                        )
                    else:
                        gt_x1, gt_x2, gt_y1, gt_y2 = -1, -1, -1, -1
                        error = [-100, -100, -100, -100]
                        iou = 0
                        iou_estimated = 0

                    single_det_error_info = [
                        eval_img['dtIds'][i],
                        eval_img['dtScores'][i],
                        matched_gt_id,
                        imgId, catId,
                        # eval_img['aRng'],
                        eval_img['maxDet'],
                        iou,
                        iou_estimated,
                        eval_img['dtIgnore'][0][i],  # ignore flag
                        *error,
                        np.mean(np.array(error)), np.max(np.array(error)),
                        det_x1, det_x2, det_y1, det_y2,
                        gt_x1, gt_x2, gt_y1, gt_y2,
                    ]
                    errors_l_r_t_b.append(single_det_error_info)

        # sorted errors_l_r_t_b based on scores
        n = len(errors_l_r_t_b)
        sorted_errors_l_r_t_b = sorted(errors_l_r_t_b, key=lambda x: x[1], reverse=True)
        matched_det_n = [x[2] != 0 for x in sorted_errors_l_r_t_b]
        tp_ratio = list(np.cumsum(np.array(matched_det_n)) / np.array(range(1, len(matched_det_n) + 1)))
        sorted_errors_l_r_t_b = [e + [r] for e, r in zip(sorted_errors_l_r_t_b, tp_ratio)]
        print(f'===================================== \n '
              f'Total {len(errors_l_r_t_b)} detections. ')
        if output_dir is not None:
            smrc.utils.generate_pkl_file(
                pkl_file_name=os.path.join(output_dir + '_cocoEval.pkl'),
                data=cocoEval
            )
            smrc.utils.generate_pkl_file(
                pkl_file_name=os.path.join(output_dir + '_errors_l_r_t_b.pkl'),
                data=sorted_errors_l_r_t_b
            )
            smrc.utils.save_multi_dimension_list_to_file(
                filename=os.path.join(output_dir + '_errors_l_r_t_b.txt'),
                list_to_save=sorted_errors_l_r_t_b
            )
