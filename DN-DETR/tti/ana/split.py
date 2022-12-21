# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from datetime import datetime

import os

from tti.ana.utils import original_args, args_base, test_one_setting
import numpy as np

# ===================
# ===================
from main import main as main_new, get_args_parser

from tti.tti_conf import LIB_ROOT_DIR
import smrc.utils


checkpoint_dir = os.path.join(LIB_ROOT_DIR, 'logs/debug/')
CheckpointDict = dict(
    Pad32FG_RandomHalf=os.path.join(
        checkpoint_dir,
        'remove/gt_only_exp-no_split-debug_gt_remove_ratio0.5-32-gt_fg_scale_fake-fg_scale_class_all_fg/checkpoint.pth'),
    # Sgdt-v1, pad32, gt_remove_ratio0.5
    Pad32FGAll=os.path.join(
        checkpoint_dir,
        'remove/gt_only_exp-no_split-32-gt_fg_scale_fake-fg_scale_class_all_fg/checkpoint.pth'),
    SGDTV1_Pad32_RandomHalf=os.path.join(
        checkpoint_dir,
        'remove/regular_5-sgdtv1_1-R50-debug_gt_remove_ratio0.5/checkpoint.pth'),

    SGDTV1_Pad16FGAll=os.path.join(
        checkpoint_dir,
        'remove/gt_only_exp-no_split-32-gt_fg_scale_fake-fg_scale_class_all_fg/checkpoint.pth'),
)


def sgdt_transformer_decoder_per_layer_analysis():





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




if __name__ == '__main__':
    sgdt_transformer_decoder_per_layer_analysis()
