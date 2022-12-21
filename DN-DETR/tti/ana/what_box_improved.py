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
from datasets.coco_eval import convert_to_xywh
import copy


LINE_THICKNESS = 2
COCO_ANNOTATION_FILE = os.path.join(LIB_ROOT_DIR, 'coco/annotations/instances_val2017.json')
RESULT_DIR = os.path.join(LIB_ROOT_DIR, 'visualize/ana_what_object_improved/full')  # v0
smrc.utils.generate_dir_if_not_exist(RESULT_DIR)

annType = 'bbox'  # specify type here
assert os.path.isfile(COCO_ANNOTATION_FILE)
cocoGt = COCO(COCO_ANNOTATION_FILE)


# class CompareDetection:
#
#     def __int__(self):


def load_and_merge_result(input_dir):
    assert os.path.isdir(input_dir)
    pred_files = smrc.utils.get_file_list_in_directory(input_dir, ext_str='.pkl')
    assert len(pred_files) > 0

    # conver the predictions to
    coco_results = []
    padded_info = {}
    for pred_file in pred_files:
        pred = torch.load(pred_file)

        coco_results.extend(prepare_for_coco_detection(pred['res']))
        padded_info.update(prepare_padded_info(pred['res']))

    #     # res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
    #     for res_info, img_info in zip(pred['res_info'], pred['img_info']):
    #         img_id = img_info['image_id']
    #
    #         for k in range(res_info.shape[0]):
    #             box = list(res_info[k, :4].numpy())
    #             score = res_info[k, 5].item()
    #             label = res_info[k, -1].item()
    #
    #             coco_results.append({
    #                         "image_id": img_id,
    #                         "category_id": label,
    #                         "bbox": box,
    #                         "score": score,
    #                     })
    return coco_results, padded_info


def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        if not isinstance(prediction["scores"], list):
            scores = prediction["scores"].tolist()
        else:
            scores = prediction["scores"]
        if not isinstance(prediction["labels"], list):
            labels = prediction["labels"].tolist()
        else:
            labels = prediction["labels"]

        try:
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        except:
            import ipdb;
            ipdb.set_trace()
    return coco_results


def prepare_padded_info(predictions):
    padded_info = {}
    for img_id, size_info in predictions.items():
        if size_info['padded_area'] == 0:
            padded_info[img_id] = ''
        else:
            padded_info[img_id] = f'_size{int(size_info["size"][0])}_' \
                                  f'{int(size_info["size"][1])}-' \
                                  f'pad_size{int(size_info["padded_img_size"][0])}_' \
                                  f'{int(size_info["padded_img_size"][1])}'
        # v["size"] = list(target["size"].cpu().numpy())
        # # v["size"] = target["size"]
        # v['padded_img_size'] = list(target['padded_img_size'].cpu().numpy())
        # v['padded_size'] = list((target['padded_img_size'] - target["size"]).cpu().numpy())
        # v['padded_area'] = (target['padded_img_size'].prod() - target["size"].prod()).cpu().item()

    return padded_info


def eval_det(pred):
    cocoDt = cocoGt.loadRes(pred)
    # imgIds = sorted(cocoGt.getImgIds())
    # imgIds=imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]

    imgIds = cocoGt.getImgIds()

    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds

    # cocoEval.params.catIds = self.cat_ids
    # cocoEval.params.maxDets = list(proposal_nums)

    # default iou threshold for tiny object detection.
    iou_thrs = np.array([0.5, 0.6])  #
    cocoEval.params.iouThrs = iou_thrs
    # cocoEval.params.maxDets = [100]
    # cocoEval.params.areaRng = [[0, 10000000000.0]]
    # # [[0, 10000000000.0], [0, 1024], [1024, 9216], [9216, 10000000000.0]]
    # cocoEval.params.areaRngLbl = ['all']  # ['all', 'small', 'medium', 'large']

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # coco_eval_bbox = cocoEval.stats.tolist()
    return cocoEval


def get_eval_result(input_dir):
    coco_res_, padded_info = load_and_merge_result(input_dir)
    return eval_det(coco_res_), padded_info


def compare_pred_of_two_methods(coco_eval_detr_original, coco_eval_improved, save_path):
    p = coco_eval_detr_original.params
    # count = 0
    img_ids = {k: [] for k in coco_eval_detr_original.cocoGt.imgs.keys()}
    det_ana_res = dict(
        improved=copy.deepcopy(img_ids),
        worse=copy.deepcopy(img_ids),
        img_paths=copy.deepcopy(img_ids),
        padded_info=copy.deepcopy(img_ids),
    )  # image id, det id, det box, matched gt id,

    gt_inds = {k: False for k in coco_eval_detr_original.cocoGt.anns.keys()}
    for k, (eval_img, eval_img_improved) in enumerate(zip(coco_eval_detr_original.evalImgs, coco_eval_improved.evalImgs)):
        # ['all', 'small', 'medium', 'large']  [1, 10, 100]
        if eval_img is None or eval_img['aRng'] != p.areaRng[0] or eval_img['maxDet'] != 100:
            continue

        if eval_img_improved is None:  # None means there is no gt.
            continue
        # if len(gt) == 0 and len(dt) ==0:
        #     return None

        assert eval_img['aRng'] == eval_img_improved['aRng'] and \
               eval_img['maxDet'] == eval_img_improved['maxDet'] and \
               eval_img['image_id'] == eval_img_improved['image_id'] and \
               eval_img['category_id'] == eval_img_improved['category_id']

        imgId, catId = eval_img['image_id'], eval_img['category_id']

        # [0] means using iou_thd = 0.5, [0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95]
        gt_matches0, gt_matches1 = eval_img['gtMatches'][0], eval_img_improved['gtMatches'][0]
        assert len(gt_matches0) == len(gt_matches1) and eval_img['gtIds'] == eval_img_improved['gtIds']

        img_path = coco_eval_detr_original.cocoGt.imgs[imgId]['file_name']
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
            # return [det_box_[0], det_box_[0] + det_box_[2], det_box_[1], det_box_[1] + det_box_[3]]
            return [det_box_[0], det_box_[1], det_box_[0] + det_box_[2], det_box_[1] + det_box_[3]]

        for i in range(len(gt_matches0)):
            gt_id = eval_img['gtIds'][i]
            gt_box = extract_box_(coco_det_=coco_eval_detr_original.cocoGt, det_id_=gt_id)

            det_id0 = int(gt_matches0[i])  # the i-th matched box
            det_id1 = int(gt_matches1[i])  # the i-th matched box

            if det_id0 == 0 and det_id1 == 0:
                continue
            else:
                det_box0 = det_box1 = None
                det_score0 = det_score1 = 0
                iou0 = iou1 = 0
                if det_id0 > 0:
                    det_box0 = extract_box_(coco_det_=coco_eval_detr_original.cocoDt,
                                            det_id_=det_id0)
                    det_index0 = eval_img['dtIds'].index(det_id0)
                    det_score0 = eval_img['dtScores'][det_index0]

                    computed_ious = coco_eval_detr_original.ious[(imgId, catId)]
                    g_id = eval_img['gtIds'].index(gt_id)
                    d_id = eval_img['dtIds'].index(det_id0)
                    iou0 = computed_ious[d_id, g_id]

                    # iou0 = smrc.utils.bbox.compute_iou(
                    #     det_box0, gt_box
                    # )

                if det_id1 > 0:
                    det_box1 = extract_box_(coco_det_=coco_eval_improved.cocoDt,
                                            det_id_=det_id1)
                    det_index1 = eval_img_improved['dtIds'].index(det_id1)
                    det_score1 = eval_img_improved['dtScores'][det_index1]

                    computed_ious = coco_eval_improved.ious[(imgId, catId)]
                    g_id = eval_img_improved['gtIds'].index(gt_id)
                    d_id = eval_img_improved['dtIds'].index(det_id1)
                    iou1 = computed_ious[d_id, g_id]

                    # iou1 = smrc.utils.bbox.compute_iou(
                    #     det_box1, gt_box
                    # )

                rec = dict(
                    image_id=imgId,
                    gt_box=gt_box,
                    gt_id=gt_id,
                    det_box0=det_box0,
                    det_score0=det_score0,
                    det_score1=det_score1,
                    det_box1=det_box1,
                    iou0=iou0,
                    iou1=iou1,
                )
                assert not(gt_inds[gt_id])
                if iou0 > iou1:
                    det_ana_res['worse'][imgId].append(rec)
                    gt_inds[gt_id] = True

                elif iou0 < iou1:
                    det_ana_res['improved'][imgId].append(rec)
                    gt_inds[gt_id] = True

    for img_id in img_ids.keys():
        if len(det_ana_res['improved'][img_id]) == 0:
            del det_ana_res['improved'][img_id]

        if len(det_ana_res['worse'][img_id]) == 0:
            del det_ana_res['worse'][img_id]

    torch.save(det_ana_res, save_path)


def extract_improved_worse(compared_result_file, batch_size,
                           padded_info_original=None,
                           padded_info_improved=None
):
    data_root_dir = os.path.join(LIB_ROOT_DIR, 'coco/val2017')
    det_ana_res_loaded = torch.load(compared_result_file)
    img_paths = det_ana_res_loaded['img_paths']
    show_iou_gap = 1e-1
    for key in ['improved', 'worse']:
        result_root_dir = os.path.join(RESULT_DIR, key + f'_batch_size{batch_size}')
        smrc.utils.generate_dir_if_not_exist(result_root_dir)
        res = det_ana_res_loaded[key]
        for img_id, box_all in res.items():
            if len(box_all) == 0:
                continue

            img_path = os.path.join(data_root_dir, img_paths[img_id])

            suffix = ''
            if padded_info_original is not None:
                suffix += f'ori_{padded_info_original[img_id]}'
            if padded_info_improved is not None:
                suffix += f'improved_{padded_info_improved[img_id]}'

            new_img_path = os.path.join(result_root_dir,
                                        smrc.utils.append_suffix_to_file_path(img_paths[img_id], suffix=suffix)
                                        )
            tmp_img = cv2.imread(img_path)

            drew = False
            for box_dict in box_all:
                gt_box, det_box0, det_box1 = box_dict['gt_box'], box_dict['det_box0'], box_dict['det_box1']
                det_score0, det_score1, iou0, iou1 = box_dict['det_score0'], box_dict['det_score1'], \
                                                     box_dict['iou0'], box_dict['iou1']
                if abs(iou0 - iou1) < show_iou_gap:
                    continue

                drew = True

                draw_single_bbox(
                    tmp_img=tmp_img, bbox=[0] + gt_box, rectangle_color=smrc.utils.YELLOW,
                )
                if det_box0 is not None:
                    draw_single_bbox(
                        tmp_img=tmp_img, bbox=[0] + det_box0, rectangle_color=smrc.utils.GREEN,
                        # caption=f'iou{smrc.utils.float_to_str(iou0, num_digits=2)}_'
                        #         f'{smrc.utils.float_to_str(det_score0, num_digits=2)}'
                    )

                if det_box1 is not None:
                    draw_single_bbox(
                        tmp_img=tmp_img, bbox=[0] + det_box1, rectangle_color=smrc.utils.RED,
                        # caption=f'iou{smrc.utils.float_to_str(iou1, num_digits=2)}_'
                        #         f'{smrc.utils.float_to_str(det_score1, num_digits=2)}'
                    )
                smrc.utils.put_text_on_image(tmp_img=tmp_img, text_content='GT: Yellow\n Original ETR: Green \n'
                                                                           'SGDT: Red')
                del gt_box, det_box0, det_box1
            if drew:
                cv2.imwrite(img=tmp_img, filename=new_img_path)


def compared(batch_size_list):

    original_detr_pred_dir_path = os.path.join(RESULT_DIR, 'coco_evaluator_original_detr')  # original_detr_coco_eval
    coco_eval_detr_original, padded_info_original = get_eval_result(original_detr_pred_dir_path)

    for b in batch_size_list:
        compared_result_file = os.path.join(RESULT_DIR, f'compared_to_batchsize{b}.pkl')
        # if not os.path.isfile(compared_result_file):
        improved_pred_dir_path = os.path.join(RESULT_DIR, f'eval_results_batchsize_{b}')  # original_detr_coco_eval
        coco_eval_improved, padded_info_improved = get_eval_result(improved_pred_dir_path)

        compare_pred_of_two_methods(coco_eval_detr_original, coco_eval_improved,
                                    compared_result_file,
                                    )
        extract_improved_worse(compared_result_file, batch_size=b,
                               padded_info_original=padded_info_original,
                               padded_info_improved=padded_info_improved
                               )


def compared_second(batch_size_list):
    original_detr_pred_dir_path = os.path.join(RESULT_DIR, 'coco_evaluator_original_detr')  # original_detr_coco_eval
    coco_eval_detr_original, padded_info_original = get_eval_result(original_detr_pred_dir_path)

    for b in batch_size_list:
        compared_result_file = os.path.join(RESULT_DIR, f'compared_to_batchsize{b}.pkl')

        if not os.path.isfile(compared_result_file):
            improved_pred_dir_path = os.path.join(RESULT_DIR, f'eval_results_batchsize_{b}')  # original_detr_coco_eval
            coco_eval_improved, padded_info_improved = get_eval_result(improved_pred_dir_path)
            compare_pred_of_two_methods(coco_eval_detr_original, coco_eval_improved,
                                        compared_result_file,
                                        )
        extract_improved_worse(compared_result_file, batch_size=b,
                               padded_info_original=padded_info_original,
                               padded_info_improved=padded_info_improved
                               )


if __name__ == "__main__":
    batch_sizes = [2, 4, 8]
    compared(batch_sizes)

