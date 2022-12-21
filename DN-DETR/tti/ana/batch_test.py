# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from datetime import datetime
import json

import random
import time
from pathlib import Path
import os, sys
from typing import Optional

from util.logger import setup_logger

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_DABDETR, build_dab_deformable_detr, \
    build_dab_deformable_detr_deformable_encoder_only, build_SGDT_DABDETR

from util.utils import clean_state_dict

# ===================
import socket
# ===================
from main import main as main_new, get_args_parser

from tti.tti_conf import LIB_ROOT_DIR
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
    # '--debug',
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
    # '--debug',
]

args_base = [
    '-m', 'sgdt_dn_dab_detr',
    '--use_dn',
    '--output_dir', 'logs/train_eval_results',
    '--coco_path', 'coco',
    '--eval',
]


def test_one_setting(args):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    coco_evaluator = main_new(parser.parse_args(args))
    mAP_results = extract_mAP(coco_evaluator)
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


def test_fg_random(with_debug=False):
    """
    token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
                    '--token_scoring_discard_split_criterion',
                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                    f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                    # f'proposal_split_thd0.0',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',
    Returns:

    """
    checkpoint_file = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg/'
        'R50/checkpoint.pth',
    )
    result_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg')

    thds = np.linspace(0, 1.0, 11)

    # proposal_split_thd = 0.8
    # thds = [1.0]
    for sampling in ['-debug_gt_split_sampling_priority', '']:

        results = {}
        results_list = []
        for thd in thds:
            token_scoring_discard_split_criterion = [
                '--token_scoring_discard_split_criterion',
                f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}'
                f'{sampling}'
            ]
            print(f'------------------------------------------------- \n'
                  f'thd = {thd} \n {token_scoring_discard_split_criterion} \n'
                  f'----------------------------------------------------\n')
            args = args_base + [
                    '--encoder_layer_config', 'regular_5-sgdt_1',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion', 'fg_scale_class_all_fg',
                    '--resume', f'{checkpoint_file}',
                    '--pad_fg_pixel', '16',
                    # '--debug',
                    # '--use_proposal',
                    # # '--proposal_saved_file', 'logs/proposals_original39.3.pth',
                    # '--proposal_saved_file', 'logs/proposals_original39.3v0.pth',
                    # '--token_adaption_visualization',
                ] + token_scoring_discard_split_criterion
            if with_debug:
                args += ['--debug', '--debug_eval_iter',  '500', ]

            mAP_results_dict, coco_eval_bbox = test_one_setting(args=args)
            results[thd] = mAP_results_dict
            results_list.append([thd] + coco_eval_bbox)

        for k, v in results.items():
            print(f' thd = {k}')
            for kk, vv in v.items():
                print(f'{kk}, {vv}')

        result_file = result_dir + f'_sampling{sampling}_with_debug{str(int(with_debug))}.txt'
        smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results_list)


def test_fg_random_use_proposal_grid_search(with_debug=False):
    """
    token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
                    '--token_scoring_discard_split_criterion',
                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                    f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                    # f'proposal_split_thd0.0',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',
    Returns:

    """

    checkpoint_file = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg/'
        'R50/checkpoint.pth',
    )
    result_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg')

    # thds = np.linspace(0, 1.0, 11)

    token_scoring_discard_split_criterion = [
        '--token_scoring_discard_split_criterion',
        f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-use_proposal'
    ]  #
    results = {}
    results_list = []
    min_split_scores = np.linspace(0.1, 0.9, 9)
    nms_thds = [1.0]
    for use_conf_score in [False]:  #   '', '-use_conf_score' , True
        for nms_thd in nms_thds:
            for min_split_score in min_split_scores[::-1]:
                print(f'------------------------------------------------- \n'
                      f'min_split_score = {min_split_score} \n {token_scoring_discard_split_criterion} \n'
                      f'----------------------------------------------------\n')
                proposal_scoring = f'min_split_score{min_split_score}-nms_thd{nms_thd}'
                if use_conf_score:
                    proposal_scoring += f'-use_conf_score'

                args = args_base + [
                        '--encoder_layer_config', 'regular_5-sgdt_1',
                        '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                        '--token_scoring_gt_criterion',  'fg_scale_class_all_fg',
                        '--resume', f'{checkpoint_file}',
                        '--pad_fg_pixel', '16',
                        # '--debug',
                        # '--use_proposal',
                        '--proposal_scoring', f'{proposal_scoring}',
                        # # '--proposal_saved_file', 'logs/proposals_original39.3.pth',
                        # '--proposal_saved_file', 'logs/proposals_original39.3v0.pth',

                        # '--token_adaption_visualization',
                    ] + token_scoring_discard_split_criterion
                if with_debug:
                    args += ['--debug', '--debug_eval_iter', '500',
                             '--proposal_saved_file',
                             # 'logs/proposals_original39.3v0.pth',
                             'logs/proposals_original44.7.pth',
                             # 'logs/proposals_original44.7-500iter.pth',
                             ]
                # else:
                #     args += ['--proposal_saved_file',
                #              # 'logs/proposals_original39.3v0.pth',
                #              'logs/proposals_original44.7.pth',
                #              ]
                mAP_results_dict, coco_eval_bbox = test_one_setting(args=args)
                # results[min_split_score] = mAP_results_dict
                results_list.append([use_conf_score, nms_thd, min_split_score] + coco_eval_bbox)
                print(f'{results_list[-1]}')
            # for k, v in results.items():
            #     print(f' thd = {k}')
            #     for kk, vv in v.items():
            #         print(f'{kk}, {vv}')

    result_file = result_dir + f'_use_proposal44.7_with_debug{str(int(with_debug))}-' \
                               f'grid-search{datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'
    smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results_list)


def test_fg_random_use_proposal(with_debug=False):
    """
    token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
                    '--token_scoring_discard_split_criterion',
                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                    f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                    # f'proposal_split_thd0.0',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',
    Returns:

    """

    checkpoint_file = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg/'
        'R50/checkpoint.pth',
    )
    result_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg')

    thds = np.linspace(0, 1.0, 11)

    # proposal_split_thd = 0.8
    # nms_thds = [0.5]
    # min_split_scores = np.linspace(0.2, 0.5, 4)
    # results = []
    # result_file = os.path.join(
    #     self.grid_search_result_dir,
    #     f'{token_scoring_gt_criterion}-pad{pad_fg_pixel}-{token_scoring_discard_split_criterion}'
    # )
    # # f'-{datetime.now().strftime("%Y%m%d-%H%M%S"),}')
    # smrc.utils.generate_dir_for_file_if_not_exist(result_file)
    #
    # for use_conf_score in [False]:  # , True
    #     for nms_thd in nms_thds:
    #         for min_split_score in min_split_scores:
    #             split_metirc, _ = self.eval_split_remove(
    #

    thds = [1.0]
    min_split_scores = np.linspace(0.2, 0.5, 4)
    for sampling in ['']:  # ['-debug_gt_split_sampling_priority', ''] '-proposal_split_gt_filter',
        results = {}
        results_list = []
        for thd in thds:
            token_scoring_discard_split_criterion = [
                '--token_scoring_discard_split_criterion',
                f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-use_proposal{sampling}'
            ]  #
            # token_scoring_discard_split_criterion = [
            #     '--token_scoring_discard_split_criterion',
            #     f'gt_only_exp-reclaim_padded_region-no_bg_token_remove'
            # ]  #
            # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
            # f'proposal_split_thd0.0',  -proposal_split_thd{0.5}
            print(f'------------------------------------------------- \n'
                  f'thd = {thd} \n {token_scoring_discard_split_criterion} \n'
                  f'----------------------------------------------------\n')
            args = args_base + [
                    '--encoder_layer_config', 'regular_5-sgdt_1',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion',  'fg_scale_class_all_fg',
                    '--resume', f'{checkpoint_file}',
                    '--pad_fg_pixel', '16',

                    # '--debug',
                    # '--use_proposal',
                    # # '--proposal_saved_file', 'logs/proposals_original39.3.pth',
                    '--proposal_saved_file', 'logs/proposals_original39.3v0.pth',
                    # '--token_adaption_visualization',
                ] + token_scoring_discard_split_criterion
            if with_debug:
                args += ['--debug', '--debug_eval_iter',  '500',
                         '--proposal_saved_file', 'logs/proposals_original44.7-500iter.pth',
                         ]
            else:
                args += ['--proposal_saved_file', 'logs/proposals_original44.7.pth',
                         ]

            mAP_results_dict, coco_eval_bbox = test_one_setting(args=args)
            results[thd] = mAP_results_dict
            results_list.append([thd] + coco_eval_bbox)

        for k, v in results.items():
            print(f' thd = {k}')
            for kk, vv in v.items():
                print(f'{kk}, {vv}')

        result_file = result_dir + f'_use_proposal_with_debug{str(int(with_debug))}-{sampling}.txt'
        smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results_list)


def test_small_use_proposal(with_debug=False):
    """
    token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
                    '--token_scoring_discard_split_criterion',
                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                    f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                    # f'proposal_split_thd0.0',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',
    Returns:

    """

    checkpoint_file = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/'
        'R50/checkpoint.pth',
    )
    result_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value')

    thds = np.linspace(0, 1.0, 11)

    # proposal_split_thd = 0.8
    thds = [1.0]
    for sampling in ['']:  # ['-debug_gt_split_sampling_priority', ''] '-proposal_split_gt_filter',
        results = {}
        results_list = []
        for thd in thds:
            token_scoring_discard_split_criterion = [
                '--token_scoring_discard_split_criterion',
                f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-use_proposal{sampling}'
            ]  #
            # token_scoring_discard_split_criterion = [
            #     '--token_scoring_discard_split_criterion',
            #     f'gt_only_exp-reclaim_padded_region-no_bg_token_remove'
            # ]  #
            # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
            # f'proposal_split_thd0.0',  -proposal_split_thd{0.5}
            print(f'------------------------------------------------- \n'
                  f'thd = {thd} \n {token_scoring_discard_split_criterion} \n'
                  f'----------------------------------------------------\n')

            args = args_base + [
                    '--encoder_layer_config', 'regular_5-sgdt_1',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion',  'significance_value',
                    '--resume', f'{checkpoint_file}',
                    '--pad_fg_pixel', '16',
                    '--proposal_scoring', 'min_fg_score0.05-min_split_score0.5-nms_thd0.6',
                    # '--debug',
                    # '--use_proposal',
                    # '--proposal_saved_file', 'logs/proposals_original44.7.pth',
                    # # '--proposal_saved_file', 'logs/proposals_original39.3.pth',
                    # '--proposal_saved_file', 'logs/proposals_original39.3v0.pth',
                    # '--token_adaption_visualization',
                ] + token_scoring_discard_split_criterion
            if with_debug:
                args += ['--debug', '--debug_eval_iter',  '500',
                         '--proposal_saved_file', 'logs/proposals_original44.7-500iter.pth',
                         ]
            else:
                args += ['--proposal_saved_file', 'logs/proposals_original44.7.pth',
                         ]

            mAP_results_dict, coco_eval_bbox = test_one_setting(args=args)
            results[thd] = mAP_results_dict
            results_list.append([thd] + coco_eval_bbox)

        for k, v in results.items():
            print(f' thd = {k}')
            for kk, vv in v.items():
                print(f'{kk}, {vv}')

        result_file = result_dir + f'_use_proposal44.7_with_debug{str(int(with_debug))}-{sampling}.txt'
        smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results_list)


def test_small_use_proposal_grid_search(with_debug=False):
    """
    token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
                    '--token_scoring_discard_split_criterion',
                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                    f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                    # f'proposal_split_thd0.0',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',
    Returns:

    """

    checkpoint_file = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/'
        'R50/checkpoint.pth',
    )
    result_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value')

    token_scoring_discard_split_criterion = [
        '--token_scoring_discard_split_criterion',
        f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-use_proposal'
    ]  #
    results = {}
    results_list = []
    min_split_scores = np.linspace(0.1, 0.9, 8)
    nms_thds = [0.5]
    for use_conf_score in [False, True]:  # '', '-use_conf_score'
        for nms_thd in nms_thds:
            for min_split_score in min_split_scores:
                print(f'------------------------------------------------- \n'
                      f'min_split_score = {min_split_score} \n {token_scoring_discard_split_criterion} \n'
                      f'----------------------------------------------------\n')
                proposal_scoring = f'min_split_score{min_split_score}-nms_thd{nms_thd}'
                if use_conf_score:
                    proposal_scoring += f'-use_conf_score'

                args = args_base + [
                    '--encoder_layer_config', 'regular_5-sgdt_1',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion', 'significance_value',  # -----------------
                    '--resume', f'{checkpoint_file}',
                    '--pad_fg_pixel', '16',
                    # '--debug',
                    # '--use_proposal',
                    '--proposal_scoring', f'{proposal_scoring}',
                    # # '--proposal_saved_file', 'logs/proposals_original39.3.pth',
                    # '--proposal_saved_file', 'logs/proposals_original39.3v0.pth',
                    # '--token_adaption_visualization',
                ] + token_scoring_discard_split_criterion
                if with_debug:
                    args += ['--debug', '--debug_eval_iter', '500',
                             '--proposal_saved_file', 'logs/proposals_original44.7-500iter.pth',
                             ]
                else:
                    args += ['--proposal_saved_file', 'logs/proposals_original44.7.pth',
                             ]

                mAP_results_dict, coco_eval_bbox = test_one_setting(args=args)
                # results[min_split_score] = mAP_results_dict
                results_list.append([use_conf_score, nms_thd, min_split_score] + coco_eval_bbox)
                print(f'{results_list[-1]}')
            # for k, v in results.items():
            #     print(f' thd = {k}')
            #     for kk, vv in v.items():
            #         print(f'{kk}, {vv}')

    result_file = result_dir + f'_use_proposal_with_debug{str(int(with_debug))}' \
                               f'-grid-search{datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'

    smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results_list)


def test_small_first_random(with_debug=False):
    """
    token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
                    '--token_scoring_discard_split_criterion',
                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                    f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                    # f'proposal_split_thd0.0',

                    # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}',
    Returns:

    """

    checkpoint_file = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/'
        'R50/checkpoint.pth',
    )
    result_dir = os.path.join(
        LIB_ROOT_DIR,
        'logs/debug/copied_benchmark/'
        'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value')

    thds = np.linspace(0, 1.0, 11)

    # proposal_split_thd = 0.8
    # thds = [1.0]
    for sampling in ['-debug_gt_split_sampling_priority', '']:
        results = {}
        results_list = []
        for thd in thds:
            token_scoring_discard_split_criterion = [
                '--token_scoring_discard_split_criterion',
                f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}'
                f'{sampling}'
            ]
            print(f'------------------------------------------------- \n'
                  f'thd = {thd} \n {token_scoring_discard_split_criterion} \n'
                  f'----------------------------------------------------\n')
            args = args_base + [
                    '--encoder_layer_config', 'regular_5-sgdt_1',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion',  'significance_value',
                    '--resume', f'{checkpoint_file}',
                    '--pad_fg_pixel', '16',
                    # '--debug',
                    # '--use_proposal',
                    # # '--proposal_saved_file', 'logs/proposals_original39.3.pth',
                    # '--proposal_saved_file', 'logs/proposals_original39.3v0.pth',
                    # '--token_adaption_visualization',
                ] + token_scoring_discard_split_criterion
            if with_debug:
                args += ['--debug', '--debug_eval_iter',  '500', ]

            mAP_results_dict, coco_eval_bbox = test_one_setting(args=args)
            results[thd] = mAP_results_dict
            results_list.append([thd] + coco_eval_bbox)

        for k, v in results.items():
            print(f' thd = {k}')
            for kk, vv in v.items():
                print(f'{kk}, {vv}')

        result_file = result_dir + f'_sampling{sampling}_with_debug{str(int(with_debug))}.txt'
        smrc.utils.save_multi_dimension_list_to_file(result_file, list_to_save=results_list)


def test():
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # -m sgdt_dn_dab_detr --output_dir logs/dn_dab_detr/tttt
    # --batch_size 2 --epochs 12 --lr_drop 11
    # --coco_path coco --use_dn
    # --lr 2.5e-5 --lr_backbone 2.5e-6
    # --encoder_layer_config regular_6
    # --token_scoring_discard_split_criterion gt_only_exp
    # --token_scoring_loss_criterion reg_sigmoid_l1_loss
    # --token_scoring_gt_criterion significance_value
    # --resume logs/R50_lr0.5_x2gpus/checkpoint.pth  --eval --save_results

    # python main.py -m sgdt_dn_dab_detr --output_dir logs/train_eval_results
    # --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1
    # --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value
    # --resume logs/debug/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/R50/checkpoint.pth
    # --eval  --pad_fg_pixel 16 --debug
    # --token_scoring_discard_split_criterion
    # gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio$thd
    thds = np.linspace(0.1, 1.0, 10)
    thds = [0.2]
    # proposal_split_thd = 0.8
    for thd in thds:
        print(f'------------------------------------------------- \n'
              f'thd = {thd} \n'
              f'----------------------------------------------------\n')
        # checkpoint_file = os.path.join(LIB_ROOT_DIR,
        #                                'logs/debug/copied_benchmark/'
        #                                'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/'
        #                                'R50/checkpoint.pth',
        #                                )
        checkpoint_file = os.path.join(
            LIB_ROOT_DIR,
            'logs/debug/copied_benchmark/'
            'gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_all_fg/ '
            'R50/checkpoint.pth',
        )
        args = parser.parse_args(
            args=
            [
                '-m', 'sgdt_dn_dab_detr',
                '--use_dn',
                '--output_dir', 'logs/train_eval_results',
                '--coco_path', 'coco',
                '--eval',
                '--encoder_layer_config', 'regular_5-sgdt_1',
                '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                '--token_scoring_gt_criterion', 'significance_value',
                '--resume', f'{checkpoint_file}',
                '--token_scoring_discard_split_criterion',
                f'gt_only_exp-reclaim_padded_region-no_bg_token_remove',
                # f'gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio{thd}-'
                # f'proposal_split_thd0.8',
                '--pad_fg_pixel', '16',
                # '--debug',
                # '--use_proposal',
                # '--proposal_saved_file', 'logs/proposals_original39.3.pth'
            ]
        )
        # if args.output_dir:
        #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        main_new(parser.parse_args(
            original_args
        ))
        # main_new(args)
        # original_args


def test_original_dn_detr(with_debug=False):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = original_44_7args  # original_args

    if with_debug:
        args += ['--debug', '--debug_eval_iter', '500', ]

    main_new(parser.parse_args(args))


if __name__ == '__main__':
    # CUDA_VISIABLE_DEVICES=2 python batch_test.py

    # test_fg_random(with_debug=True)
    # test_fg_random(with_debug=False)
    # test_small_first_random(with_debug=False)
    # cuda 1, with gt scoring, cuda 2, with proposal scoring
    # test_small_use_proposal(with_debug=True)
    # # test_small_use_proposal(with_debug=False)
    # test_fg_random_use_proposal(with_debug=True)
    # test_fg_random_use_proposal(with_debug=False)

    # test_original_dn_detr(with_debug=True)
    # test_fg_random_use_proposal_grid_search(with_debug=True)
    # test_fg_random_use_proposal_grid_search(with_debug=False)

    # test_small_use_proposal_grid_search(with_debug=True)
    # test_small_use_proposal_grid_search(with_debug=False)
    # test_small_use_proposal(with_debug=True)
    # test_small_use_proposal(with_debug=False)

    test_original_dn_detr(with_debug=True)
    # test_original_dn_detr(with_debug=False)