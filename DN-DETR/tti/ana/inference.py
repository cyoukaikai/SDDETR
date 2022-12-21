import argparse
from tti.tti_conf import LIB_ROOT_DIR
import os
from main import get_args_parser, main as main_new

from tti.tti_conf import LIB_ROOT_DIR
import smrc.utils


checkpoint_dir = os.path.join(LIB_ROOT_DIR, 'logs/debug/copied_benchmark')
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

checkpoint_file = CheckpointDict['Pad32FG_RandomHalf']
original_args = [
    '-m', 'sgdt_dn_dab_detr',
    '--use_dn',
    '--output_dir', os.path.join(LIB_ROOT_DIR, 'logs/train_eval_results'),
    '--coco_path', os.path.join(LIB_ROOT_DIR, 'coco'),
    '--eval',
    '--encoder_layer_config', 'regular_5-sgdt_1',
    '--resume', f'{checkpoint_file}',
    '--token_scoring_discard_split_criterion', 'gt_only_exp-debug_gt_remove_ratio0.5',
    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
    '--token_scoring_gt_criterion', 'fg_scale_class_all_fg',
]


def test(with_debug=True):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = original_args

    if with_debug:
        args += ['--debug', '--debug_eval_iter', '500', ]

    main_new(parser.parse_args(args))


if __name__ == '__main__':
    test()