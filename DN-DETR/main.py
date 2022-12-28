# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
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
from models import build_DABDETR, build_SGDT_DABDETR  #,\
    # build_dab_deformable_detr, \
    # build_dab_deformable_detr_deformable_encoder_only,  build_token_classifier

from util.utils import clean_state_dict

# ===================
import socket
from tti.tti_conf import LIB_ROOT_DIR
from models.sgdt.sgdt_ import GTRatioOrSigma
# from trained_models import get_trained_models
# import mmcv
from mmcv.runner.checkpoint import load_checkpoint, load_state_dict


# ===================


def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)

    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--modelname', '-m', type=str, required=True,
                        choices=['dn_dab_detr', 'dn_dab_deformable_detr',
                                 'dn_dab_deformable_detr_deformable_encoder_only',
                                 'sgdt_dn_dab_detr',
                                 'sgdt_token_fg_bg_classifier',
                                 ])
    # parser.add_argument('--modelname', '-m', type=str, required=True, choices=['dn_dab_detr', 'dn_dab_deformable_detr',
    #                                                                 'dn_dab_deformable_detr_deformable_encoder_only'])
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int,
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int,
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str,
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'],
                        help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str,
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true',
                        help="Using pre-norm in the Transformer blocks.")
    parser.add_argument('--num_select', default=300, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int,
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true',
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int,
                        help="number of deformable attention sampling points in encoder layers")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float,
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float,
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float,
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float,
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float,
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help="alpha for focal loss")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true',
                        help="Using for debug only. It will fix the size of input images to the maximum.")

    # Traing utils
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+',
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true',
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true',
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    # ------------------------------------------------------
    # TTI modification on the encoder
    # ------------------------------------------------------
    # parser.add_argument('--num_encoder_sgdt_layers', default=0, type=int,
    #                     help="Number of SGDT encoding layers in the transformer")
    # parser.add_argument('--encoder_sgdt_layer_version', default=0, type=int,
    #                     help="Number of SGDT encoding layers in the transformer")
    parser.add_argument('--exp_str', default='')
    parser.add_argument('--debug_eval_iter', default=0, type=int)  # 500
    parser.add_argument('--no_resume_optimizer_lr_schedule', action='store_true')
    parser.add_argument('--align_encoder_decoder_layers_num', default=0, type=int)  # 500
    parser.add_argument('--use_proposal', action='store_true')
    parser.add_argument('--proposal_scoring', default=None, type=str)
    parser.add_argument('--load_masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--freeze_weight_keywords', nargs="+", type=str, help='freeze some layers in network.')

    parser.add_argument('--freeze_weight_ignore_keywords', nargs="+", type=str)

    parser.add_argument('--loss_disable_ignore_keywords', nargs="+", type=str)
    parser.add_argument('--training_only_distill_student_attn', action='store_true')
    parser.add_argument('--training_only_distill_student_attn_not_free_backbone', action='store_true')
    parser.add_argument('--ignore_detr_loss', action='store_true')
    parser.add_argument('--freeze_sgdt_transformer_trained_layers', action='store_true')
    parser.add_argument('--freeze_transformer_sgdt_encoder_layer_ffn_out', action='store_true')
    parser.add_argument('--freeze_transformer_sgdt_encoder_layer_MHA_out', action='store_true')
    parser.add_argument('--freeze_transformer_sgdt_encoder_layer_attn_softmax_out', action='store_true')
    parser.add_argument('--freeze_online_encoder_distillation', action='store_true')
    parser.add_argument('--freeze_attn_online_encoder_distillation', action='store_true')
    parser.add_argument('--freeze_detr_decoder', action='store_true')
    parser.add_argument('--skip_teacher_model_decoder_forward', action='store_true')
    parser.add_argument('--training_skip_forward_decoder', action='store_true')
    parser.add_argument('--with_teacher_model', action='store_true', help="")
    parser.add_argument('--teacher_model_use_pretrained_sgdt_V_marked', action='store_true')
    parser.add_argument('--teacher_model_use_pretrained_detr44AP', action='store_true')
    parser.add_argument('--teacher_model_use_pretrained_v_shared_double_attn52_2ap', action='store_true')
    parser.add_argument('--teacher_model_use_pretrained_v_shared_double_attn52_6ap_lr1_not_converged', action='store_true')
    parser.add_argument('--teacher_model_use_pretrained_v_shared_double_attn52_6ap', action='store_true')
    parser.add_argument('--teacher_model_use_pretrained_attn_learning_model', action='store_true')
    parser.add_argument('--auxiliary_fg_bg_cls_encoder_layer_ids', nargs="+", type=int)

    parser.add_argument('--decoder_ca_attn_distill', default=None, type=str,)  # first_decoder_ca,
    parser.add_argument('--decoder_prediction_distill', default=None, type=str, )  # from_second_decoder, only_last_decoder

    parser.add_argument('--feature_attn_distillation', default=None, type=str,
                        help="Conduct feature_attn_distillation")  # cascade parallel separate_trained_model
    parser.add_argument('--feature_distillation_teacher_feat_with_grad', action='store_true')
    parser.add_argument('--sgdt_transformer', action='store_true',
                        help="Using the transformer that the last encoder layer accept input from decoder output")
    parser.add_argument('--dual_attn_transformer', action='store_true')

    parser.add_argument('--online_self_distill_transformer', action='store_true',)
    parser.add_argument('--online_decoder_self_distill_transformer', action='store_true',)
    parser.add_argument('--double_head_transformer', action='store_true',)
    parser.add_argument('--share_double_head_transformer', action='store_true',)
    parser.add_argument('--share_decoder_ca_attn_map', action='store_true',)

    parser.add_argument('--save_coco_evaluator_prefix', default=None, type=str,
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--train_token_scoring_only', action='store_true',
                        help="Train token scoring only in the training.")
    parser.add_argument('--decay_sigma', action='store_true',
                        help="Decay the sigma value for token selection to reduce the train test gap.")
    parser.add_argument('--eval_decoder_layer', default=-1, type=int)
    parser.add_argument('--use_decoder_proposal', action='store_true')
    parser.add_argument('--use_pretrained_model_proposal', action='store_true')
    # parser.add_argument('--marking_encoder_feature_by_fg1_bg0', action='store_true')  # deprecate in future
    parser.add_argument('--marking_encoder_layer_fg1_bg0', type=str, default='',
                        choices=['K', 'Q', 'V', 'FFN_Out', ])
    parser.add_argument('--token_classifier', action='store_true')
    # parser.add_argument('--token_classifier', type=str, default='',
    #                     choices=['Pretrained_MLP_Fg_KV_to_MLP_Classifier'])

    parser.add_argument('--lr_sgdt_transformer_trained_layers', action='store_true')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--proposal_saved_file', default=None)
    parser.add_argument('--eval_training_data', action='store_true')
    parser.add_argument('--wandb', action='store_true')

    parser.add_argument('--with_decoder_prediction_distillation', action='store_true')
    parser.add_argument('--decoder_prediction_distillation_config', default=None, type=str, required=False,
                        # choices=['last_layer', ]
                        )
    parser.add_argument('--sgdt_decoder_pred_cls_loss_coef', default=2.0, type=float,)
    parser.add_argument('--sgdt_decoder_pred_loc_loss_coef', default=5.0, type=float,)
    parser.add_argument('--with_sgdt_attention_loss', action='store_true')

    parser.add_argument('--sgdt_attention_loss_coef', default=5.0, type=float,
                        help="Attention loss")  # 0.1
    parser.add_argument('--attention_loss_top_100_token', action='store_true')
    parser.add_argument('--with_sgdt_transformer_feature_distill_loss', action='store_true')
    parser.add_argument('--sgdt_transformer_feature_distillation_loss_coef', default=5.0, type=float,
                        help="Transformer feature distillation loss")  # 0.1
    parser.add_argument('--randomly_mark_fg1_bg0_prob', default=None, type=float, help="")
    parser.add_argument('--eval_randomly_mark_fg1_bg0', default='no_mark', type=str, help="")

    parser.add_argument('--attn_distillation_teacher_with_grad', action='store_true')
    parser.add_argument('--teacher_layer_back_propagate_to_input', action='store_true')



    parser.add_argument('--debug_st_attn_sweep_n_attn_heads', type=int, default=0,)
    parser.add_argument('--debug_st_attn_sweep_fg_attn', action='store_true')
    parser.add_argument('--debug_st_attn_sweep_bg_attn', action='store_true')

    parser.add_argument('--sgdt_loss_token_significance_coef', default=1.0, type=float,
                        help="down weight of the sgdt loss")
    parser.add_argument('--sgdt_loss_fg_coef', default=1.0, type=float,
                        help="down weight of the sgdt loss")
    parser.add_argument('--sgdt_loss_small_scale_coef', default=1.0, type=float,
                        help="down weight of the sgdt loss")
    parser.add_argument('--visualization_out_sub_dir', default=None, type=str)
    parser.add_argument('--token_adaption_visualization', action='store_true')
    parser.add_argument('--attention_map_evaluation', default=None, type=str)  # action='store_true'

    parser.add_argument('--disable_fg_scale_supervision', action='store_true')
    parser.add_argument('--encoder_without_pos', action='store_true')
    parser.add_argument('--pad_fg_pixel', type=int, default=0, help='The number of pixels in input image to extend'
                                                                     'from box boundary as gt regions.')
    parser.add_argument('--encoder_layer_config', default='regular_6', required=False,
                        )  # 'regular_6',  'regular_4-sgdtv1_1-sgdt_1', '' for TransformerEmptyEncoder
    parser.add_argument('--decoder_layer_config', default='regular_6', required=False,
                        )  # 'regular_6',  'regular_4-sgdtv1_1-sgdt_1', '' for TransformerEmptyEncoder
    parser.add_argument('--proposal_token_scoring_gt_criterion', type=str, default='confidence_value',
                        required=False)
    parser.add_argument('--gt_decay_criterion', type=str, default=None)  # 'start_epoch8-end_epoch11'
    parser.add_argument('--token_scoring_gt_criterion', type=str, default='', required=False,
                        choices=['significance_value',
                                 # 'significance_value_bg_w_priority',
                                 'significance_value_inverse_fg',
                                 'significance_value_from_instance_mask',
                                 'fg_scale_class_small_medium_random',
                                 'fg_scale_class_small_random',
                                 'fg_scale_class_all_fg',
                                 'fake_all_tokens_are_fg',
                                 'fg_scale_class_all_bg',
                                 ]
                        )
    parser.add_argument('--token_scoring_loss_criterion', type=str, required=False,
                        default='',
                        choices=['reg_sigmoid_l1_loss',
                                 'gt_fg_scale_fake',
                                 'fg_scale_class_pred_focal_loss',
                                 'fg_scale_class_pred_ce_independent',
                                 'fg_weighted_ce',
                                 ]
                        )
    parser.add_argument('--token_scoring_discard_split_criterion', type=str, required=False,
                        default='', )
    # -----------------
    return parser


def build_model_main(args):
    if args.modelname.lower() == 'dn_dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    # elif args.modelname.lower() == 'dn_dab_deformable_detr':
    #     model, criterion, postprocessors = build_dab_deformable_detr(args)
    # elif args.modelname.lower() == 'dn_dab_deformable_detr_deformable_encoder_only':
    #     model, criterion, postprocessors = build_dab_deformable_detr_deformable_encoder_only(args)
    elif args.modelname.lower() == 'sgdt_dn_dab_detr':
        model, criterion, postprocessors = build_SGDT_DABDETR(args)
    # elif args.modelname.lower() == 'sgdt_token_fg_bg_classifier':
    #     model, criterion, postprocessors = build_token_classifier(args)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    # torch.autograd.set_detect_anomaly(True)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False,
                          name="DAB-DETR")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: " + ' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config.json")
        # print("args:", vars(args))
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    #######################
    if args.auto_resume:
        assert args.resume == ''
        resume_checkpoint_file = os.path.join(args.output_dir, 'checkpoint.pth')
        if os.path.isfile(resume_checkpoint_file):
            args.resume = resume_checkpoint_file
        else:
            args.resume = ''

    if args.resume and os.path.isfile(args.resume):
        print(f'Not set the random seed, resume from {args.resume}')
    else:
        # fix the seed for reproducibility
        print(f'Set the random seed to {args.seed}')
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    # -------------

    # build model
    model, criterion, postprocessors = build_model_main(args)

    freeze_weight_keywords = []
    if args.freeze_sgdt_transformer_trained_layers:
        """
          "module.class_embed.weight": 23296,
          "module.class_embed.bias": 91,
          "module.label_enc.weight": 23460,
          "module.refpoint_embed.weight": 1200,
        """
        freeze_weight_keywords = ['transformer.encoder.layers.5', 'transformer.decoder.layers.5',
                                  'label_enc', 'refpoint_embed', 'class_embed']
        # for param_name, param in model.named_parameters():
        #     if param_name.find('transformer.encoder.layers.5') > -1 or \
        #             param_name.find('transformer.decoder.layers.5') > -1 or \
        #             param_name.find('label_enc') > -1 or param_name.find('refpoint_embed') > -1 or \
        #             param_name.find('class_embed') > -1:
        #         print(f'{param_name}, requires_grad = {param.requires_grad}')
        #     else:
        #         param.requires_grad = False
    elif args.freeze_transformer_sgdt_encoder_layer_ffn_out:
        freeze_weight_keywords = ['transformer.encoder', 'backbone', 'input_proj', 'sgdt']
    elif args.freeze_transformer_sgdt_encoder_layer_MHA_out:
        """
        transformer.encoder.layers.5.self_attn.in_proj_weight
        transformer.encoder.layers.5.self_attn.in_proj_bias
        transformer.encoder.layers.5.self_attn.out_proj.weight
        transformer.encoder.layers.5.self_attn.out_proj.bias
        transformer.encoder.layers.5.linear1.weight
        transformer.encoder.layers.5.linear1.bias
        transformer.encoder.layers.5.linear2.weight
        transformer.encoder.layers.5.linear2.bias
        transformer.encoder.layers.5.norm1.weight
        transformer.encoder.layers.5.norm1.bias
        transformer.encoder.layers.5.norm2.weight
        transformer.encoder.layers.5.norm2.bias
        transformer.encoder.layers.5.activation.weight
        """
        # args.freeze_transformer_sgdt_encoder_layer_MHA_out:
        # weights to initialize ('transformer.encoder.layers.5.norm2', 'transformer.encoder.layers.5.linear',
        # 'transformer.encoder.layers.5.activation',
        # )
        freeze_weight_keywords = ['transformer.encoder.query_scale',
                                  'transformer.encoder.layers.0', 'transformer.encoder.layers.1',
                                  'transformer.encoder.layers.2', 'transformer.encoder.layers.3',
                                  'transformer.encoder.layers.4',
                                  'transformer.encoder.layers.5.self_attn',
                                  'transformer.encoder.layers.5.norm1',
                                  'backbone', 'input_proj', 'sgdt']
    elif args.freeze_transformer_sgdt_encoder_layer_attn_softmax_out:
        # only freeze w_k, w_q
        freeze_weight_keywords = ['transformer.encoder.query_scale',
                                  'transformer.encoder.layers.0', 'transformer.encoder.layers.1',
                                  'transformer.encoder.layers.2', 'transformer.encoder.layers.3',
                                  'transformer.encoder.layers.4',
                                  # 'transformer.encoder.layers.5.self_attn.in_proj',  # packed wk, wq, wv
                                  'backbone', 'input_proj', 'sgdt']
    elif args.freeze_online_encoder_distillation:
        """
            "module.transformer.encoder.layers.5.self_attn.out_proj_teacher.
            "module.transformer.encoder.layers.5.teacher_encoder_layer.,
            "module.sgdt.sgdt_module.token_split_conv.linear.0.weight": 65536,
            "module.sgdt.sgdt_module.token_split_conv.linear.0.bias": 256
        """
        freeze_weight_keywords = ['transformer.decoder.layers.5',  # freeze teacher decoder branch
                                  'teacher', 'sgdt']
    elif args.freeze_attn_online_encoder_distillation:
        """
            "module.transformer.encoder.layers.5.self_attn.out_proj_teacher.
            "module.transformer.encoder.layers.5.teacher_encoder_layer.,
            "module.sgdt.sgdt_module.token_split_conv.linear.0.weight": 65536,
            "module.sgdt.sgdt_module.token_split_conv.linear.0.bias": 256
        """
        # freeze_weight_keywords = ['backbone', 'input_proj',
        #                           'transformer.encoder.layers.0',
        #                           'transformer.encoder.layers.1',
        #                           'transformer.encoder.layers.2',
        #                           'transformer.encoder.layers.3',
        #                           'transformer.encoder.layers.4',
        #
        #                           'transformer.decoder.layers.5',
        #                           'teacher', 'sgdt']

        freeze_weight_keywords = ['sgdt']

    elif args.freeze_detr_decoder:
        freeze_weight_keywords = ['transformer.decoder.',
                                  'label_enc', 'refpoint_embed', 'class_embed']

    if len(freeze_weight_keywords) == 0:
        freeze_weight_keywords = args.freeze_weight_keywords if args.freeze_weight_keywords else []

    if len(freeze_weight_keywords) > 0:
        # #######################################
        # # disable the training of encoder and backbone, only train the Decoder.
        # for param_name, param in model.named_parameters():
        #     if param_name.find('transformer.encoder') > -1 or \
        #             param_name.find('backbone') > -1 or \
        #             param_name.find('input_proj') > -1:
        #         param.requires_grad = False
        # #######################################
        # freeze_weight_keywords = args.freeze_weight_keywords if args.freeze_weight_keywords else []
        print('===========================================')
        for param_name, param in model.named_parameters():
            for keywords in freeze_weight_keywords:
                if param_name.find(keywords) > -1:
                    param.requires_grad = False
                    print(f'{param_name}, requires_grad = {param.requires_grad}')
                    break
        print('===========================================')
    elif args.train_token_scoring_only:
        # freeze everything except the parameters for top-k prediction
        for param_name, param in model.named_parameters():
            # the gradient of decoder is needed for propagating gradient back to the scoring layer.
            if param_name.find('token_scoring') > -1 or param_name.find('transformer.decoder') > -1 or \
                    param_name.find('label_enc') > -1 or param_name.find('refpoint_embed') > -1 or \
                    param_name.find('class_embed') > -1:
                print(f'{param_name}, requires_grad = {param.requires_grad}')
            else:
                param.requires_grad = False
                print(f'{param_name}, requires_grad = {param.requires_grad}')

            # if param_name.find('token_scoring') > -1 or param_name.find('transformer.decoder') > -1:
            #     print(f'{param_name}, requires_grad = {param.requires_grad}')
            # else:
            #     param.requires_grad = False

    freeze_weight_ignore_keywords = []
    if args.training_only_distill_student_attn:
        # if args.freeze_weight_ignore_keywords:
        # freeze_weight_ignore_keywords = args.freeze_weight_ignore_keywords if args.freeze_weight_ignore_keywords else []
        """
          "module.transformer.encoder.layers.5.self_attn.in_proj_weight": 327680,
          "module.transformer.encoder.layers.5.self_attn.in_proj_bias": 1280,
          "module.transformer.encoder.layers.5.self_attn.out_proj.weight": 65536,
          "module.transformer.encoder.layers.5.self_attn.out_proj.bias": 256,
        """
        # current setting regular_4-parallelSTECSGDTShareVOutProjFFN_1
        freeze_weight_ignore_keywords = ['transformer.encoder.layers.4.self_attn.in_proj']

    elif args.training_only_distill_student_attn_not_free_backbone:
        freeze_weight_ignore_keywords = [
            'transformer.encoder.layers.4.self_attn.in_proj',
            'input_proj',
            'backbone',

            'transformer.encoder.query_scale',
            'transformer.encoder.layers.0.',
            'transformer.encoder.layers.1.',
            'transformer.encoder.layers.2.',
            'transformer.encoder.layers.3.',
            # 'sgdt.sgdt_module.',
        ]

    if len(freeze_weight_ignore_keywords) > 0:
        print('===========================================')
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:' + str(n_parameters))
        for param_name, param in model.named_parameters():
            freeze_flag = True
            for keywords in freeze_weight_ignore_keywords:
                if param_name.find(keywords) > -1:
                    freeze_flag = False
                    break

            if freeze_flag:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f'{param_name}, requires_grad = {param.requires_grad}')
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:' + str(n_parameters))
        print('===========================================')

    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:' + str(n_parameters))
    logger.info(
        "params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    if args.lr_sgdt_transformer_trained_layers:
        """
          "module.class_embed.weight": 23296,
          "module.class_embed.bias": 91,
          "module.label_enc.weight": 23460,
          "module.refpoint_embed.weight": 1200,
        """
        params_normal_lr = []
        params_special_lr = []
        for n, p in model_without_ddp.named_parameters():
            if not p.requires_grad: continue

            # below are the layers to train.
            if (n.find('transformer.encoder.layers.5') > -1 or n.find('transformer.decoder.layers.5') > -1 or
                    n.find('label_enc') > -1 or n.find('refpoint_embed') > -1 or
                    n.find('class_embed') > -1):
                params_normal_lr.append(p)
            else:  # other layers use the same lr as the lr_backbone.
                params_special_lr.append(p)

        param_dicts = [
            {"params": params_normal_lr},
            {"params": params_special_lr, "lr": args.lr_backbone}
        ]
    elif args.train_token_scoring_only:
        params_normal_lr = []
        # params_special_lr = []
        for n, p in model_without_ddp.named_parameters():
            if not p.requires_grad: continue

            # do not update the parameters for other parameters..
            if n.find('token_scoring') > -1:
                params_normal_lr.append(p)

        param_dicts = [
            {"params": params_normal_lr},
            # {"params": params_special_lr, "lr": args.lr_backbone}
        ]

    else:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', args=args)

    if args.eval_training_data:  # Evaluate the training dataset.
        # simply set 'train' will use the pre-defined training transform, but we need the val transform (random resize
        # only)
        dataset_val = build_dataset(image_set='train_val', args=args)
    else:
        dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    data_size = len(data_loader_train.batch_sampler)  # batch_sampler, 59143, single GPU, batch size 2;  dataset: 118286
    gt_ratio_or_sigma = GTRatioOrSigma(
        gt_decay_criterion=args.gt_decay_criterion,
        data_size=data_size, total_epoch=args.epochs,
        decay_sigma=args.decay_sigma,
    )

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        # load_state_dict(model_without_ddp, checkpoint['model'])

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            # ========================
            # If we continue for x2 schedule, the setting will continue to use the x1 schedule.
            # So we deprecate the following original setting.

            # ======================================== (update the lr based on resumed epoch)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, last_epoch=checkpoint['epoch'])

            # No load the old optimizer
            # if not args.no_resume_optimizer_lr_schedule:
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # =================================
            args.start_epoch = checkpoint['epoch'] + 1

            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']  # OrderedDict

        from collections import OrderedDict

        _ignorekeywordlist = []
        if args.freeze_transformer_sgdt_encoder_layer_ffn_out:
            _ignorekeywordlist = ['decoder']
        elif args.freeze_transformer_sgdt_encoder_layer_MHA_out:
            # weights to not initialize with pre-trained model
            _ignorekeywordlist = ['decoder', 'transformer.encoder.layers.5.norm2',
                                  'transformer.encoder.layers.5.linear',
                                  'transformer.encoder.layers.5.activation']
        elif args.freeze_transformer_sgdt_encoder_layer_attn_softmax_out:
            """
            transformer.encoder.layers.5.self_attn.in_proj_weight
            transformer.encoder.layers.5.self_attn.in_proj_bias
            
            transformer.encoder.layers.5.self_attn.out_proj.weight
            transformer.encoder.layers.5.self_attn.out_proj.bias
            transformer.encoder.layers.5.linear1.weight
            transformer.encoder.layers.5.linear1.bias
            transformer.encoder.layers.5.linear2.weight
            transformer.encoder.layers.5.linear2.bias
            transformer.encoder.layers.5.norm1.weight
            transformer.encoder.layers.5.norm1.bias
            transformer.encoder.layers.5.norm2.weight
            transformer.encoder.layers.5.norm2.bias
            transformer.encoder.layers.5.activation.weight
            """
            _ignorekeywordlist = ['decoder',
                                  # the self_attn.in_proj_weight and in_proj_bias will be manually loaded later, so
                                  # they are ignored here
                                  'transformer.encoder.layers.5.self_attn',
                                  'transformer.encoder.layers.5.linear',  # linear1 linear2
                                  'transformer.encoder.layers.5.norm',  # norm1 norm2
                                  'transformer.encoder.layers.5.activation',
                                  ]
            # only freeze w_k, w_q

        if len(_ignorekeywordlist) == 0:
            _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []

        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {k: v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)

        # manually load the weights of w_q, w_k, b_q, b_k, do not load w_v, b_v
        if args.freeze_transformer_sgdt_encoder_layer_attn_softmax_out:
            device = model_without_ddp.transformer.encoder.layers[5].self_attn.in_proj_weight.data.device
            w_q, w_k, w_v = checkpoint['transformer.encoder.layers.5.self_attn.in_proj_weight'].to(device).chunk(3)
            b_q, b_k, b_v = checkpoint['transformer.encoder.layers.5.self_attn.in_proj_bias'].to(device).chunk(3)

            w_q_o, w_k_o, w_v_o = model_without_ddp.transformer.encoder.layers[5].self_attn.in_proj_weight.data.chunk(3)
            b_q_o, b_k_o, b_v_o = model_without_ddp.transformer.encoder.layers[5].self_attn.in_proj_bias.data.chunk(3)

            w_new = torch.cat([w_q, w_k, w_v_o])
            b_new = torch.cat([b_q, b_k, b_v_o])

            model_without_ddp.transformer.encoder.layers[5].self_attn.in_proj_weight.data = w_new
            model_without_ddp.transformer.encoder.layers[5].self_attn.in_proj_bias.data = b_new

            # model_without_ddp.transformer.encoder.layers[5].self_attn.in_proj_weight.data =
            # checkpoint['transformer.encoder.layers.5.self_attn.in_proj_weight'].chunk(3)

        logger.info(str(_load_output))
        # import ipdb; ipdb.set_trace()

    def load_pretrain_model_(args_pretrained_model_):
        device_ = torch.device(args_pretrained_model_.device)
        # build model
        model_, _, _ = build_model_main(args_pretrained_model_)
        # wo_class_error = False
        model_.to(device_)

        model_without_ddp_ = model_
        if args.distributed:  # modified ------------------------------
            model_ = torch.nn.parallel.DistributedDataParallel(
                model_, device_ids=[args.gpu],
                find_unused_parameters=args.find_unused_params)
            model_without_ddp_ = model_.module
        # ------------------------------------------------------
        if args_pretrained_model_.frozen_weights is not None:
            checkpoint_ = torch.load(args_pretrained_model_.frozen_weights, map_location='cpu')
            model_without_ddp_.detr.load_state_dict(checkpoint_['model'])

        # output_dir = Path(args.output_dir)
        if args_pretrained_model_.resume:
            if args_pretrained_model_.resume.startswith('https'):
                checkpoint_ = torch.hub.load_state_dict_from_url(
                    args_pretrained_model_.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint_ = torch.load(args_pretrained_model_.resume, map_location='cpu')
            model_without_ddp_.load_state_dict(checkpoint_['model'])

        if not args_pretrained_model_.resume and args_pretrained_model_.pretrain_model_path:
            checkpoint_ = torch.load(args_pretrained_model_.pretrain_model_path, map_location='cpu')['model']

            from collections import OrderedDict

            _ignorekeywordlist = args_pretrained_model_.finetune_ignore if args_pretrained_model_.finetune_ignore else []
            ignorelist = []

            def check_keep(keyname, ignorekeywordlist):
                for keyword in ignorekeywordlist:
                    if keyword in keyname:
                        ignorelist.append(keyname)
                        return False
                return True

            # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
            _tmp_st = OrderedDict(
                {k: v for k, v in clean_state_dict(checkpoint_).items() if check_keep(k, _ignorekeywordlist)})
            _load_output = model_without_ddp_.load_state_dict(_tmp_st, strict=False)
            # logger.info(str(_load_output))

        print('Loading trained model done.')
        for param in model_.parameters():
            param.requires_grad = False
        model_.eval()
        return model_

    def get_trained_models():
        proposal_parser = argparse.ArgumentParser('', parents=[get_args_parser()])
        args_pretrained_model = proposal_parser.parse_args(
            args=
            [
                '-m', 'sgdt_dn_dab_detr',
                '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
                '--use_dn',  # previous version, I forgot this.
                '--encoder_layer_config', 'regular_6',
                # '--resume', f'{os.path.join(LIB_ROOT_DIR, "logs/R50_lr0.5_x2gpus/checkpoint.pth")}',
                '--resume', f'{os.path.join(LIB_ROOT_DIR, "logs/checkpoint_optimized_44.7ap.pth")}',
            ]
        )
        return load_pretrain_model_(args_pretrained_model)

    def get_distillation_pretrained_models():
        """
            pad_fg_pixel=0
            token_scoring_loss_criterion=gt_fg_scale_fake
            token_scoring_gt_criterion=significance_value
            token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
            out_dir=regular_5-sgdtv1_1
            exp_str=feature-distillation-new-split1c_version-gt_split_only-aligh-sgdtv1_1
            #===========================

            python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
            main.py -m sgdt_dn_dab_detr \
              --output_dir logs/$out_dir/$exp_str \
              --exp_str  $exp_str \
              --batch_size 2 \
              --epochs 12 \
              --lr_drop 11 \
              --coco_path coco \
              --use_dn \
              --lr 5e-5 \
              --lr_backbone 5e-6 \
              --encoder_layer_config regular_5-sgdtv1_1 \
              --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
              --token_scoring_loss_criterion $token_scoring_loss_criterion  \
              --token_scoring_gt_criterion $token_scoring_gt_criterion \
              --pad_fg_pixel $pad_fg_pixel \
              --align_encoder_decoder_layers_num 1 \
              --auto_resume \
              --feature_attn_distillation \
              --save_checkpoint_interval 1 \
              --wandb
        Returns:

        """
        proposal_parser = argparse.ArgumentParser('', parents=[get_args_parser()])
        # out_dir_ = 'logs/regular_5-sgdtv1_1/token_num_no_limit-aligh-sgdtv1_1-debug_split_1c'
        # args_pretrained_model = proposal_parser.parse_args(
        #     args=
        #     [
        #         '-m', 'sgdt_dn_dab_detr',
        #         '--output_dir', f'{out_dir_}',
        #         '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
        #         '--use_dn',
        #         # 'eval',
        #         '--encoder_layer_config', 'regular_5-sgdtv1_1',
        #         '--pretrain_model_path', f'{os.path.join(out_dir_, "checkpoint.pth")}',
        #         '--token_scoring_discard_split_criterion', 'gt_only_exp-no_bg_token_remove',
        #         '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
        #         '--token_scoring_gt_criterion', 'significance_value',
        #         '--pad_fg_pixel', '0',
        #         '--align_encoder_decoder_layers_num', '1',
        #     ]
        # )
        if args.teacher_model_use_pretrained_sgdt_V_marked:
            out_dir_ = 'logs/e6-d6-gt_split_only/e6-d6-V-gt_split_only-regular_5-sgdt+v_1'
            args_pretrained_model = proposal_parser.parse_args(
                args=
                [
                    '-m', 'sgdt_dn_dab_detr',
                    '--output_dir', f'{out_dir_}',
                    '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
                    '--use_dn',
                    # 'eval',
                    '--encoder_layer_config', 'regular_5-sgdt+v_1',
                    '--pretrain_model_path', f'{os.path.join(out_dir_, "checkpoint.pth")}',
                    '--token_scoring_discard_split_criterion', 'gt_only_exp-no_bg_token_remove',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion', 'significance_value',
                    '--pad_fg_pixel', '0',
                    # '--align_encoder_decoder_layers_num', '1',
                ]
            )

        elif args.teacher_model_use_pretrained_detr44AP:
            out_dir_ = 'logs'
            args_pretrained_model = proposal_parser.parse_args(
                args=
                [
                    '-m', 'sgdt_dn_dab_detr',
                    '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
                    '--use_dn',
                    '--output_dir', f'{out_dir_}',
                    '--encoder_layer_config', 'regular_6',
                    '--token_scoring_gt_criterion', 'significance_value',
                    '--pad_fg_pixel', '0',
                    '--resume', f'{os.path.join(LIB_ROOT_DIR, "logs/checkpoint_optimized_44.7ap.pth")}',
                ]
            )
        elif args.teacher_model_use_pretrained_v_shared_double_attn52_2ap or \
                args.teacher_model_use_pretrained_v_shared_double_attn52_6ap or \
                args.teacher_model_use_pretrained_v_shared_double_attn52_6ap_lr1_not_converged:
            """

                    # to start from epoch 10
                    pad_fg_pixel=0
                    token_scoring_loss_criterion=gt_fg_scale_fake
                    token_scoring_gt_criterion=significance_value
                    token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
                    out_dir=e6-d6-gt_split_only
                    encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
                    exp_str=share_double_head_transformer_ShareV_out_proj_FFN
                    #===========================

                    python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
                    main.py -m sgdt_dn_dab_detr \
                      --output_dir logs/$out_dir/$exp_str \
                      --exp_str  $exp_str \
                      --batch_size 2 \
                      --epochs 50 \
                      --lr_drop 40 \
                      --drop_lr_now \
                      --coco_path coco \
                      --use_dn \
                      --lr 5e-5 \
                      --lr_backbone 5e-6 \
                      --encoder_layer_config $encoder_layer_config \
                      --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
                      --token_scoring_loss_criterion $token_scoring_loss_criterion  \
                      --token_scoring_gt_criterion $token_scoring_gt_criterion \
                      --pad_fg_pixel $pad_fg_pixel \
                      --share_double_head_transformer \
                      --eval_decoder_layer 3 \
                      --decoder_layer_config regular_4 \
                      --auto_resume \
                      --no_resume_optimizer_lr_schedule \
                      --save_checkpoint_interval 2 \
                      --wandb
                    """
            if args.teacher_model_use_pretrained_v_shared_double_attn52_6ap:
                out_dir_ = 'logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN'
            elif args.teacher_model_use_pretrained_v_shared_double_attn52_6ap_lr1_not_converged:
                out_dir_ = 'logs/e6-d6-gt_split_only/teacher_model_use_pretrained_v_shared_double_attn52_6ap_lr1_not_converged'
            else:
                out_dir_ = 'logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN_lr0.1_from_epoch35'
            print(f'out_dir_ = {out_dir_}')
            args_pretrained_model = proposal_parser.parse_args(
                args=
                [
                    '-m', 'sgdt_dn_dab_detr',
                    '--output_dir', f'{out_dir_}',
                    '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
                    '--use_dn',
                    # 'eval',
                    '--encoder_layer_config', 'regular_4-parallelSTECSGDTShareVOutProjFFN_1',
                    '--pretrain_model_path', f'{os.path.join(out_dir_, "checkpoint.pth")}',
                    '--token_scoring_discard_split_criterion', 'gt_only_exp-no_bg_token_remove',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion', 'significance_value',
                    '--pad_fg_pixel', '0',
                    # '--eval_decoder_layer', '3',
                    '--share_double_head_transformer',
                    '--decoder_layer_config', 'regular_4',
                ]
            )

        elif args.teacher_model_use_pretrained_attn_learning_model:
            """
            encoder_layer_config=regular_6
            exp_str=ST_attention_learning_no_decoder_from_sMLP_Fg_KV51AP
            #===========================
            
            python main.py -m sgdt_dn_dab_detr \
              --output_dir logs/$out_dir/$exp_str \
              --exp_str  $exp_str \
              --batch_size 2 \
              --epochs 8 \
              --lr_drop 5 \
              --coco_path coco \
              --use_dn \
              --lr 5e-5 \
              --lr_backbone 5e-6 \
              --encoder_layer_config $encoder_layer_config \
              --token_scoring_gt_criterion $token_scoring_gt_criterion \
              --pad_fg_pixel $pad_fg_pixel \
              --with_teacher_model \
              --skip_teacher_model_decoder_forward \
              --with_sgdt_attention_loss \
              --sgdt_attention_loss_coef 20 \
              --training_skip_forward_decoder \
              --loss_disable_ignore_keywords attention feature \
              --ignore_detr_loss \
              --freeze_detr_decoder \
              --with_sgdt_transformer_feature_distill_loss \
              --sgdt_transformer_feature_distillation_loss_coef 1 \
              --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
              --wandb   
            """
            out_dir_ = 'logs/e6-d6-gt_split_only/ST_attention_learning_no_decoder_from_sMLP_Fg_KV51AP'
            args_pretrained_model = proposal_parser.parse_args(
                args=
                [
                    '-m', 'sgdt_dn_dab_detr',
                    '--output_dir', f'{out_dir_}',
                    '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
                    '--use_dn',
                    # 'eval',
                    '--encoder_layer_config', 'regular_6',
                    '--pretrain_model_path', f'{os.path.join(out_dir_, "checkpoint.pth")}',
                    '--token_scoring_discard_split_criterion', 'gt_only_exp-no_bg_token_remove',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion', 'significance_value',
                    '--pad_fg_pixel', '0',
                    '--loss_disable_ignore_keywords', 'attention feature',
                    '--training_skip_forward_decoder',
                    '--ignore_detr_loss',
                    '--freeze_detr_decoder',
                ]
            )
        else:
            out_dir_ = 'logs/e6-d6-gt_split_only/gt_split_only-regular_5-sgdtv1_1-e6-d6'
            args_pretrained_model = proposal_parser.parse_args(
                args=
                [
                    '-m', 'sgdt_dn_dab_detr',
                    '--output_dir', f'{out_dir_}',
                    '--coco_path', f'{os.path.join(LIB_ROOT_DIR, "coco")}',
                    '--use_dn',
                    # 'eval',
                    '--encoder_layer_config', 'regular_5-sgdtv1_1',
                    '--pretrain_model_path', f'{os.path.join(out_dir_, "checkpoint.pth")}',
                    '--token_scoring_discard_split_criterion', 'gt_only_exp-no_bg_token_remove',
                    '--token_scoring_loss_criterion', 'gt_fg_scale_fake',
                    '--token_scoring_gt_criterion', 'significance_value',
                    '--pad_fg_pixel', '0',
                    # '--align_encoder_decoder_layers_num', '1',
                ]
            )
        return load_pretrain_model_(args_pretrained_model)

    # if args.token_scoring_discard_split_criterion.find('use_proposal') > -1:  # args.use_proposal? --
    proposal_model = None
    if args.use_pretrained_model_proposal:
        proposal_model = get_trained_models()

    teacher_model = None
    if args.feature_attn_distillation or args.with_teacher_model:
        teacher_model = get_distillation_pretrained_models()

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        eval_decoder_layers = [args.eval_decoder_layer]
        # if args.eval_decoder_layer is not None:
        #     assert isinstance(args.eval_decoder_layer, int)
        #     if args.eval_decoder_layer != -1:
        #         eval_decoder_layers = [args.eval_decoder_layer, -1]
        #     else:
        #         eval_decoder_layers = [-1]

        coco_evaluator = None
        for eval_decoder_layer in eval_decoder_layers:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir,
                                                  wo_class_error=wo_class_error,
                                                  args=args,
                                                  proposal_model=proposal_model,
                                                  teacher_model=teacher_model,
                                                  eval_decoder_layer=eval_decoder_layer,
                                                  epoch=args.start_epoch
                                                  # proposal_processor=model.sgdt.proposal_processor
                                                  )
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / f"eval-{eval_decoder_layer}.pth")

            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
            if args.output_dir and utils.is_main_process():
                if eval_decoder_layer == -1:
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    with (output_dir / f"log-{eval_decoder_layer}.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            # if args.output_dir and utils.is_main_process():
            #     with (output_dir / "log.txt").open("a") as f:
            #         f.write(json.dumps(log_stats) + "\n")

        return coco_evaluator

    # print("Start training")
    # ---------------------------# Weights & Biases
    if args.wandb and args.local_rank == 0:  # args.local_rank == 0, os.environ.get('LOCAL_RANK', -1) == 0
        import wandb  # not here to set
        # import mmcv
        # from mmcv.runner import get_dist_info, init_dist
        # # re-set gpu_ids with distributed training mode
        # _, world_size = get_dist_info()
        if args.exp_str != '':
            exp_prefix = f'{socket.gethostname()}_{args.exp_str}_{args.encoder_layer_config}'
        else:
            exp_prefix = f'{socket.gethostname()}_{args.modelname.lower()}_bs' \
                         f'{args.batch_size}x{args.world_size}_lr' \
                         f'{args.lr}_{args.encoder_layer_config}_{args.token_scoring_discard_split_criterion}'
        # args.world_size   --encoder_layer_config regular_4-sgdtv1_1-sgdt_1  --token_scoring_version v0_with_gt
        # https://docs.wandb.ai/guides/track/advanced/resuming
        # store this id to use it later when resuming
        id = wandb.util.generate_id()
        # wandb.init(id=id, resume="allow")
        wandb.init(project="SGDT", entity="kaikaizhao", name=exp_prefix,
                   id=id, resume="allow",
                   # resume=True
                   )
        # https://docs.wandb.ai/guides/track/advanced/resuming#:~:text=Resume%20Runs%20%2D%20Documentation&text=
        # You%20can%20have%20wandb%20automatically,logging%20from%20the%20last%20step.
    else:
        wandb = None
        # ---------------------------
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error,
            lr_scheduler=lr_scheduler, args=args,
            # logger=(logger if args.save_log else None),
            wandb=wandb,  # --------
            gt_ratio_or_sigma=gt_ratio_or_sigma,
            proposal_model=proposal_model,
            teacher_model=teacher_model,
            # proposal_processor=model.sgdt.proposal_processor
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_beforedrop.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if args.eval_decoder_layer is not None:
            assert isinstance(args.eval_decoder_layer, int)
            if args.eval_decoder_layer != -1:
                eval_decoder_layers = [args.eval_decoder_layer, -1]
            else:
                eval_decoder_layers = [-1]

            for eval_decoder_layer in eval_decoder_layers:
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args,
                    logger=(logger if args.save_log else None),
                    # --------
                    wandb=wandb,
                    proposal_model=proposal_model,
                    teacher_model=teacher_model,
                    eval_decoder_layer=eval_decoder_layer,
                    epoch=epoch,
                    # proposal_processor=model.sgdt.proposal_processor
                )

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': v for k, v in test_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                epoch_time = time.time() - epoch_start_time
                epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
                log_stats['epoch_time'] = epoch_time_str

                if args.output_dir and utils.is_main_process():
                    if eval_decoder_layer == -1:
                        with (output_dir / "log.txt").open("a") as f:
                            f.write(json.dumps(log_stats) + "\n")
                    else:
                        with (output_dir / f"log-{eval_decoder_layer}.txt").open("a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                    # for evaluation logs
                    if coco_evaluator is not None:
                        (output_dir / 'eval').mkdir(exist_ok=True)
                        if "bbox" in coco_evaluator.coco_eval:
                            filenames = ['latest.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                           output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
