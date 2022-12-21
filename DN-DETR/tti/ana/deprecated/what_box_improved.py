from tti.tti_conf import LIB_ROOT_DIR
import smrc.utils
import sys
import os
import os.path
import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tti.draw_utils import draw_single_bbox
# def ana_improved_det(cocoEval, cocoEvalImproved):
#     det_ana_res = dict(
#         improved=[],
#         worse=[],
#     )  # image id, det id, det box, matched gt id,
#     count = 0
#     for k, (eval_img, eval_img_improved) in enumerate(zip(cocoEval.evalImgs, cocoEvalImproved.evalImgs)):
#         for i in range(len(eval_img['dtIds'])):
#
#             det_id = eval_img['dtIds'][i]
#             det_box = cocoEval.cocoDt.anns[det_id]['bbox']
#             # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2
#             det_x1, det_x2, det_y1, det_y2 = [det_box[0], det_box[0] + det_box[2],
#                                               det_box[1], det_box[1] + det_box[3]]
#             # # =============== iou related section
#             # # skip ignored detection
#             # if eval_img['dtIgnore'][0][i]:
#             #     continue
#
#             matched_gt_id = eval_img['dtMatches'][0][i]  # iou thd = 0.05
#             # ===============
#             imgId, catId = eval_img['image_id'], eval_img['category_id'],
#             if matched_gt_id != 0:  # matched with gt
#                 gt_box = cocoEval.cocoGt.anns[matched_gt_id]['bbox']
#                 gt_x1, gt_x2, gt_y1, gt_y2 = [gt_box[0], gt_box[0] + gt_box[2],
#                                               gt_box[1], gt_box[1] + gt_box[3]]
#                 error_abs = list(np.abs(np.array([det_x1, det_x2, det_y1, det_y2]) -
#                                         np.array([gt_x1, gt_x2, gt_y1, gt_y2])))
#                 error = np.array([det_x1, det_x2, det_y1, det_y2]) - np.array([gt_x1, gt_x2, gt_y1, gt_y2])
#
#                 # # The saved iou has different order
#                 computed_ious = cocoEval.ious[(imgId, catId)]
#                 g_id = eval_img['gtIds'].index(matched_gt_id)
#                 iou = computed_ious[i, g_id]
#                 if iou < iou_thd:
#                     continue
#
#                 if eval_img['dtScores'][i] < score_thd:
#                     continue
#                 # iou_estimated = smrc.utils.bbox.compute_iou(
#                 #     [det_x1, det_y1, det_x2, det_y2],
#                 #     [gt_x1, gt_y1, gt_x2, gt_y2]
#                 # )
#                 #
#                 # modify the prediction
#                 # det_box = cocoEval.cocoDt.anns[det_id]['bbox']
#                 # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2
#
#                 count += 1
#                 det_x1_new, det_x2_new, det_y1_new, det_y2_new = \
#                     np.array([det_x1, det_x2, det_y1, det_y2]) - error * ratio
#
#                 cocoEval.cocoDt.anns[det_id]['bbox'] = [det_x1_new, det_y1_new,
#                                                         det_x2_new - det_x1_new,
#                                                         det_y2_new - det_y1_new
#                                                         ]
#                 single_det_error_info = [
#                     eval_img['dtIds'][i],
#                     eval_img['dtScores'][i],
#                     np.mean(np.array(error_abs)), np.max(np.array(error_abs)),
#                     *error_abs,
#                     matched_gt_id,
#                     iou,
#                     eval_img['dtIgnore'][0][i],  # ignore flag
#
#                 ]
#                 det_ana_res.append(single_det_error_info)
#             # else:
#             #     print('not matched')
#             #     # gt_x1, gt_x2, gt_y1, gt_y2 = -1, -1, -1, -1
#             #     # error = [-100, -100, -100, -100]
#             #     # iou = 0
#             #     # iou_estimated = 0
#
#             # single_det_error_info = [
#             #     eval_img['dtIds'][i],
#             #     eval_img['dtScores'][i],
#             #     matched_gt_id,
#             #     imgId, catId,
#             #     # eval_img['aRng'],
#             #     eval_img['maxDet'],
#             #     iou,
#             #     iou_estimated,
#             #     eval_img['dtIgnore'][0][i],  # ignore flag
#             #     *error,
#             #     np.mean(np.array(error)), np.max(np.array(error)),
#             #     det_x1, det_x2, det_y1, det_y2,
#             #     gt_x1, gt_x2, gt_y1, gt_y2,
#             # ]
#             # errors_l_r_t_b.append(single_det_error_info)
#
#     #
#     # count = 0
#     # result = []
#     # for k, eval_img in enumerate(cocoEval.evalImgs):  # loop through every gt box
#     #     # Only process false positive examples
#     #     for i in range(len(eval_img['dtIds'])):  # det are sorted by scores from high to low
#     #         # matched with gt or det with low confidence
#     #         if eval_img['dtMatches'][0][i] != 0 or eval_img['dtScores'][i] < score_thd:
#     #             continue
#     #         else:
#     #             det_id = eval_img['dtIds'][i]
#     #             det_box = cocoEval.cocoDt.anns[det_id]['bbox']
#     #
#     #             # cat ID of gt box also means the cat ID of det box
#     #             imgId, catId = eval_img['image_id'], eval_img['category_id']
#     #
#     #             det_info = {
#     #                 'id': count,
#     #                 'image_id': imgId,
#     #                 'bbox': det_box,
#     #                 'area': det_box[2] * det_box[3],
#     #                 'category_id': catId,
#     #                 # "iscrowd": 0,
#     #             }
#     #             count += 1
#     #             result.append(det_info)
#     #
#     #             if i >= max_det:
#     #                 break
#
#     print(f'{count} hard negative detections saved for {len(cocoEval.evalImgs)} images')
#     return result
LINE_THICKNESS = 2

COCO_ANNOTATION_FILE = os.path.join(LIB_ROOT_DIR, 'coco/annotations/instances_val2017.json')
RESULT_DIR = os.path.join(LIB_ROOT_DIR, 'visualize/ana_what_object_improved')
smrc.utils.generate_dir_if_not_exist(RESULT_DIR)

#
original_detr_pred_file = os.path.join(RESULT_DIR, 'coco_evaluator_original_detr.pkl')  # original_detr_coco_eval
pred_file = os.path.join(RESULT_DIR, 'coco_evaluator_batchsize_2.pkl')

# coco_eval_detr_original = smrc.utils.load_pkl_file(original_detr_pred_file)
# coco_eval = smrc.utils.load_pkl_file(pred_file)

coco_eval_detr_original = torch.load(original_detr_pred_file).coco_eval['bbox']
coco_eval_improved = torch.load(pred_file).coco_eval['bbox']
# ana_improved_det(cocoEval=coco_eval_detr_original, cocoEvalImproved=coco_eval_improved)

p = coco_eval_detr_original.params
count = 0
img_ids = {k: [] for k in coco_eval_detr_original.evalImgs.cocoGt.imgs.keys()}
det_ana_res = dict(improved=img_ids.copy(), worse=img_ids.copy(),
                   img_paths=img_ids.copy())  # image id, det id, det box, matched gt id,
for k, (eval_img, eval_img_improved) in enumerate(zip(coco_eval_detr_original.evalImgs, coco_eval_improved.evalImgs)):
    # ['all', 'small', 'medium', 'large']
    if eval_img is None or eval_img['aRng'] != p.areaRng[0] or eval_img['maxDet'] != 1000:
        continue

    imgId, catId = eval_img['image_id'], eval_img['category_id']

    # [0] means using iou_thd = 0.5, [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
    gt_matches0, gt_matches1 = eval_img['gtMatches'][0], eval_img_improved['gtMatches'][0]
    assert len(gt_matches0) == len(gt_matches1) and eval_img['gtIds'] == eval_img_improved['gtIds']

    img_path = coco_eval_detr_original.cocoGt['imgs'][imgId]['file_name']
    det_ana_res['img_paths'][imgId] = img_path


    def extract_box_(coco_det_, det_id_):
        """
             det_box = coco_eval_detr_original.cocoDt.anns[det_id]['bbox']  # keys > 0
            # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2
            det_x1, det_x2, det_y1, det_y2 = [det_box[0], det_box[0] + det_box[2],
                                              det_box[1], det_box[1] + det_box[3]]
        Args:
            coco_det_:
            det_id_:
        Returns:

        """
        det_box_ = coco_det_.anns[det_id_]['bbox']
        # det_box[2:] = det_box[:2] + det_box[2:]  # x1, y1, w, h -> x1, y1, x2, y2
        return [det_box_[0], det_box_[0] + det_box_[2], det_box_[1], det_box_[1] + det_box_[3]]


    # num_non_zeros = min(np.count_nonzero(gt_matches0), np.count_nonzero(gt_matches0))
    for i in range(len(gt_matches0)):
        gt_id = eval_img['gtIds'][i]
        gt_box = extract_box_(coco_det_=coco_eval_detr_original.cocoGt, det_id_=gt_id)

        det_id0 = gt_matches0[i]  # the i-th matched box
        det_id1 = gt_matches1[i]  # the i-th matched box

        if det_id0 == 0 and det_id1 == 0:
            continue
        else:
            det_box0 = det_box1 = None
            det_score0 = det_score1 = 0
            iou0 = iou1 = 0
            if det_id0 > 0:
                det_box0 = extract_box_(coco_det_=coco_eval_detr_original.cocoDt,
                                        det_id_=eval_img['dtIds'][det_id0])
                det_score0 = eval_img['dtScores'][det_id0]

                computed_ious = coco_eval_detr_original.ious[(imgId, catId)]
                g_id = eval_img['gtIds'].index(gt_id)
                # d_id = eval_img['dtIds'].index(det_id0)
                iou = computed_ious[det_id0, g_id]

                iou0 = smrc.utils.bbox.compute_iou(
                    det_box0, gt_box
                )

            if det_id1 > 0:
                det_box1 = extract_box_(coco_det_=coco_eval_improved.cocoDt,
                                        det_id_=eval_img_improved['dtIds'][det_id1])
                det_score1 = eval_img_improved['dtScores'][det_id1]

                computed_ious = coco_eval_improved.ious[(imgId, catId)]
                g_id = eval_img['gtIds'].index(gt_id)
                # d_id = eval_img['dtIds'].index(det_id0)
                iou0 = computed_ious[det_id0, g_id]

                iou1 = smrc.utils.bbox.compute_iou(
                    det_box1, gt_box
                )

            rec = dict(
                image_id=imgId,
                gt_box=gt_box,
                det_box0=det_box0,
                det_score0=det_score0,
                det_score1=det_score1,
                det_box1=det_box1,
                iou0=iou0,
                iou1=iou1,
            )

            if iou0 > iou1:
                det_ana_res['worse'][imgId].append(rec)
            elif iou1 > iou0:
                det_ana_res['improved'][imgId].append(rec)

save_path = 'compared_to_batchsize2.pkl'
torch.save(det_ana_res, save_path)

data_root_dir = os.path.join(LIB_ROOT_DIR, 'coco/val2017')

det_ana_res_loaded = torch.load(save_path)
img_paths = det_ana_res['img_paths']

for key in ['improved', 'worse']:
    result_root_dir = os.path.join(RESULT_DIR, key)
    smrc.utils.generate_dir_if_not_exist(result_root_dir)
    for res in det_ana_res[key]:

        for img_id, box_all in res.items():

            img_path = os.path.join(data_root_dir, img_paths[img_id])
            new_img_path = os.path.join(result_root_dir, img_paths[img_id])
            tmp_img = cv2.imread(img_path)

            for box_dict in box_all:
                gt_box, det_box0, det_box1 = box_dict['gt_box'], box_dict['det_box0'], box_dict['det_box1']
                det_score0, det_score1, iou0, iou1 = box_dict['det_score0'], box_dict['det_score1'],\
                                                     box_dict['iou0'], box_dict['iou1']


                draw_single_bbox(
                    tmp_img=tmp_img, bbox=gt_box, rectangle_color=smrc.utils.YELLOW,
                )
                if det_box0 is not None:
                    draw_single_bbox(
                        tmp_img=tmp_img, bbox=det_box0, rectangle_color=smrc.utils.GREEN,
                    )

                if det_box0 is not None:
                    draw_single_bbox(
                        tmp_img=tmp_img, bbox=det_box0, rectangle_color=smrc.utils.GREEN,
                        caption=f'iou{}_{det_score1}'
                    )
            cv2.imwrite(img=tmp_img, filename=new_img_path)





#     # remove_ana = RemoveAnalysis()
#     # remove_ana.easy_remove_ana(with_debug=True)
