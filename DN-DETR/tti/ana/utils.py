import argparse
from tti.tti_conf import LIB_ROOT_DIR
import os
from main import get_args_parser, main as main_new
import smrc.utils


original_args = [
    '-m', 'sgdt_dn_dab_detr',
    '--coco_path', 'coco',
    '--use_dn',
    '--output_dir', 'logs/train_eval_results',
    '--eval',
    '--encoder_layer_config', 'regular_6',
    '--resume', 'logs/R50_lr0.5_x2gpus/checkpoint.pth',
    '--token_scoring_discard_split_criterion', 'gt_only_exp',
    '--token_scoring_loss_criterion', 'reg_sigmoid_l1_loss',
    '--token_scoring_gt_criterion', 'significance_value',
]
original_44_7args = [
    '-m', 'sgdt_dn_dab_detr',
    '--coco_path', 'coco',
    '--use_dn',
    '--output_dir', 'logs/train_eval_results',
    '--eval',
    '--encoder_layer_config', 'regular_6',
    '--resume', 'logs/checkpoint_optimized_44.7ap.pth',  # R50_lr0.5_x2gpus/checkpoint.pth
    '--token_scoring_discard_split_criterion', 'gt_only_exp',
    '--token_scoring_loss_criterion', 'reg_sigmoid_l1_loss',
    '--token_scoring_gt_criterion', 'significance_value',

]
args_base = [
    '-m', 'sgdt_dn_dab_detr',
    '--use_dn',
    '--output_dir', os.path.join(LIB_ROOT_DIR, 'logs/train_eval_results'),
    '--coco_path', os.path.join(LIB_ROOT_DIR, 'coco'),
    '--eval',
]


def test_one_setting(args, eval_result_file=None):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    coco_evaluator = main_new(parser.parse_args(args))
    mAP_results = extract_mAP(coco_evaluator)

    if eval_result_file is not None and eval_result_file != '':
        save_coco_eval(coco_eval=coco_evaluator, save_path=eval_result_file)
    return mAP_results


def extract_mAP(coco_evaluator):
    results = {}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            coco_eval_bbox = coco_evaluator.coco_eval['bbox'].stats.tolist()
            metric_name = [
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
                'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
                'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]'
            ]
            results = {k: v for k, v in zip(metric_name, coco_eval_bbox)}
    return results, coco_eval_bbox


def test_original_dn_detr(with_debug=False):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # args = original_44_7args  # original_args
    args = original_args
    if with_debug:
        args += ['--debug', '--debug_eval_iter', '500', ]
    test_one_setting(args, eval_result_file=os.path.join(LIB_ROOT_DIR, 'original_detr_coco_eval.pkl'))
    # main_new(parser.parse_args(args))



def save_coco_eval(coco_eval, save_path):
    smrc.utils.generate_pkl_file(pkl_file_name=save_path, data=coco_eval)



if __name__ == '__main__':
    test_original_dn_detr()