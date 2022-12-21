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
from models import build_DABDETR, build_dab_deformable_detr, \
    build_dab_deformable_detr_deformable_encoder_only, build_SGDT_DABDETR

from util.utils import clean_state_dict

# ===================
import socket
from trained_models import get_trained_models
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
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str, 
                        help='freeze some layers in backbone. for catdet5.')
    # parser.add_argument('--enc_layers', default=6, type=int,
    #                     help="Number of encoding layers in the transformer")
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
    parser.add_argument('--use_proposal', action='store_true')
    parser.add_argument('--proposal_saved_file', default=None)
    parser.add_argument('--eval_training_data', action='store_true')
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--sgdt_loss_token_significance_coef', default=1.0, type=float,
                        help="down weight of the sgdt loss")
    parser.add_argument('--sgdt_loss_fg_coef', default=1.0, type=float,
                        help="down weight of the sgdt loss")
    parser.add_argument('--sgdt_loss_small_scale_coef', default=1.0, type=float,
                        help="down weight of the sgdt loss")
    # parser.add_argument('--sgdt_loss_fg_coef', default=1.0, type=float,
    #                     help="down weight of the sgdt loss")  # 0.1

    parser.add_argument('--token_adaption_visualization', action='store_true')
    parser.add_argument('--disable_fg_scale_supervision', action='store_true')
    parser.add_argument('--encoder_without_pos', action='store_true')
    # padded value should be large than 16.
    parser.add_argument('--pad_fg_pixel', type=int, default=16, help='The number of pixels in input image to extend'
                                                                     'from box boundary as gt regions.')

    parser.add_argument('--encoder_layer_config', default='regular_6', required=True,
                        )  # 'regular_6',  'regular_4-sgdtv1_1-sgdt_1', '' for TransformerEmptyEncoder
    # parser.add_argument('--reclaim_padded_region', action='store_true')
    parser.add_argument('--token_scoring_gt_criterion', type=str, default='significance_value',
                        choices=['significance_value',
                                 # 'significance_value_bg_w_priority',
                                 'significance_value_inverse_fg',
                                 'fg_scale_class_small_medium_random',
                                 'fg_scale_class_all_fg',
                                 'fake_all_tokens_are_fg'
                                 ]
                        )
    parser.add_argument('--token_scoring_loss_criterion', type=str, required=True,
                        default='reg_sigmoid_l1_loss',
                        choices=['reg_sigmoid_l1_loss',
                                 'gt_fg_scale_fake',
                                 'fg_scale_class_pred_focal_loss',
                                 'fg_scale_class_pred_ce_independent',
                                 'fg_weighted_ce',
                                 ]
                        )  # L1Loss
    parser.add_argument('--token_scoring_discard_split_criterion', type=str, required=True,  # '-m',
                        default='gt_only_exp',
                        # choices=[
                        # gt_only_exp # remove and split 'v0_with_gt',
                        # gt_only_exp-reclaim_padded_region # 'v0_with_gt_and_reclaim_padded_region',
                        # gt_only_exp-reclaim_padded_region-no_bg_token_remove # 'v0_with_gt_only_reclaim_padded_region'
                        # gt_only_exp-no_split  # 'v0_with_gt_only_remove',  # remove only

                        # pred_significance-reclaim_padded_region-no_bg_token_remove
                        # pred_significance-split_sig_thd0.7-bg_sig_thd0.3

                        #          'test_but_scoring_grad_for_loss_only',
                        #          'v1_selection_differentiable_sig_value',
                        #          'v2_selection_by_pred_label_no_grad',

                        #          'pred_significance_all_fg_w_priority',
                        #          'pred_significance_all_fg_w_priority_only_remove',
                        #          'pred_significance_all_fg_w_priority_only_reclaim_padded_region',
                        #          'pred_significance_all_fg_w_priority_and_reclaim_padded_region'
                        #          'pred_significance_all_fg_bg_w_priority',
                        #          'pred_fg_only_remove',
                        #          'pred_fg_only_remove_0.05',
                        #          'pred_fg_only_remove_0.1_wo_gf',
                        #          'pred_fg_only_remove_0.3_wo_gf',
                        #          'pred_fg_only_remove_0.5_wo_gf',
                        #          'pred_fg_only_remove_0.1_w_gf',
                        #
                        #
                        #          'dynamic_vit_pred_fg_only_remove_0.1_w_gf',
                        #          'dynamic_vit_pred_fg_only_remove_0.1_wo_gf',
                        #          'conv_pred_fg_only_remove_0.1',
                        #          'conv_pred_fg_only_remove_0.3',
                        #          'conv_pred_fg_only_remove_0.5',
                        #
                        #          'conv_pred_fg_only_remove_min_thd1.0_debug',
                        # 'conv_pred_fg_only_remove-min_thd0.5-inverse-debug',
                        #          ]
                        )
    # -----------------


    return parser


def build_model_main(args):
    if args.modelname.lower() == 'dn_dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    elif args.modelname.lower() == 'sgdt_dn_dab_detr':
        model, criterion, postprocessors = build_SGDT_DABDETR(args)
    elif args.modelname.lower() == 'dn_dab_deformable_detr':
        model, criterion, postprocessors = build_dab_deformable_detr(args)
    elif args.modelname.lower() == 'dn_dab_deformable_detr_deformable_encoder_only':
        model, criterion, postprocessors = build_dab_deformable_detr_deformable_encoder_only(args)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)
    # torch.autograd.set_detect_anomaly(True)
    
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['output_dir'] = args.output_dir
    # logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="DAB-DETR")
    # logger.info("git:\n  {}\n".format(utils.get_sha()))
    # logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config.json")
        # print("args:", vars(args))
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        # logger.info("Full config saved to {}".format(save_json_path))
    # logger.info('world size: {}'.format(args.world_size))
    # logger.info('rank: {}'.format(args.rank))
    # logger.info('local_rank: {}'.format(args.local_rank))
    # logger.info("args: " + str(args) + '\n')

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info('number of params:'+str(n_parameters))
    # logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))


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
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        # logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        # logger.info(str(_load_output))
        # import ipdb; ipdb.set_trace()

    if args.use_proposal:
        proposal_model = get_trained_models()
    else:
        proposal_model = None

    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir,
                                              wo_class_error=wo_class_error, args=args,
                                              proposal_model=proposal_model)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # print("Start training")

    # ---------------------------# Weights & Biases
    # os.environ['LOCAL_RANK']
    if args.wandb and os.environ.get('LOCAL_RANK', -1) == 0:  # args.local_rank == 0
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
        wandb.init(project="SGDT", entity="kaikaizhao",  name=exp_prefix,
                   # resume=True
                   )
        # https://docs.wandb.ai/guides/track/advanced/resuming#:~:text=Resume%20Runs%20%2D%20Documentation&text=You%20can%20have%20wandb%20automatically,logging%20from%20the%20last%20step.
        # wandb.config = {
        #     "learning_rate": args.lr,
        #     "epochs": args.epochs,
        #     "batch_size": args.batch_size,
        #     "num_workers": args.num_workers
        # }
        # # wandb.log({"loss": loss})
        # # Optional
        # wandb.watch(model)
    else:
        wandb = None
        # ---------------------------

    start_time = time.time()

    # for p in model.parameters():
    #     if p.grad is None:
    #         print("found unused param")

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
        
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None),
            wandb=wandb,  # --------
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
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
    # print('Training time {}'.format(total_time_str))
    # print("Now time: {}".format(str(datetime.datetime.now())))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
