# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from util.utils import to_device


from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

# ========================
import smrc.utils
from models.sgdt.scoring_gt import sgdt_update_sample_target, update_targets_with_proposals
from models.sgdt.sgdt_components import is_with_sgdt_layer
from models.sgdt.sgdt_ import init_proposal_processor

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


import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset

from util.utils import clean_state_dict

# ===================
import socket
from tti.tti_conf import LIB_ROOT_DIR
from models.sgdt.sgdt_ import GTRatioOrSigma


# ===================
from main import get_args_parser, build_model_main

from engine import train_one_epoch  # evaluate,


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             wo_class_error=False, args=None, logger=None,
             wandb=None,
             proposal_model=None,
             feature_distillation_teacher_model=None,
             ):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    proposal_processor = init_proposal_processor(args.proposal_scoring)

    proposal_to_save = {}
    proposal_saved_file = args.proposal_saved_file
    if proposal_model is not None or args.use_decoder_proposal:
        if proposal_saved_file is not None and os.path.isfile(proposal_saved_file):
            proposals_loaded = smrc.utils.load_pkl_file(proposal_saved_file)
        else:
            proposals_loaded = None
        # proposal_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    _cnt = 0
    output_state_dict = {}  # for debug only
    output_fg_attn = []
    break_iter = 15  # default setting
    if args.debug_eval_iter > 0:
        break_iter = args.debug_eval_iter

    # cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, 20, header, logger=logger):
        # if cnt > 5:
        #     break
        # else:
        #     cnt += 1

        # ------------------- tti
        samples, targets = sgdt_update_sample_target(samples, targets, args)
        # -------------------
        # image_ids = [t['image_id'].item() for t in targets]
        # print(f'img_ids = {image_ids}')
        # # if 3845 not in image_ids:
        # #     continue

        samples = samples.to(device)
        # import ipdb; ipdb.set_trace()
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        if proposal_model is not None:
            if proposals_loaded is not None:
                pred_boxes = []
                pred_logits = []
                for t in targets:
                    pred = proposals_loaded[t['image_id'].cpu().item()]
                    pred_boxes.append(pred['pred_boxes'])
                    pred_logits.append(pred['pred_logits'])
                proposals = {
                    'pred_boxes': torch.stack(pred_boxes, dim=0).to(samples.device),
                    'pred_logits': torch.stack(pred_logits, dim=0).to(samples.device),
                }
            else:
                with torch.cuda.amp.autocast(enabled=args.amp):
                    if need_tgt_for_training:
                        proposals, _ = proposal_model(samples, dn_args=args.num_patterns,
                                                      sgdt_args=(targets,)
                                                      )
                    else:
                        proposals = proposal_model(samples)

                for kk, t in enumerate(targets):
                    proposal_pred = {
                        'pred_boxes': proposals['pred_boxes'][kk].cpu(),
                        'pred_logits': proposals['pred_logits'][kk].cpu()
                    }
                    proposal_to_save.update(
                        {
                            t['image_id'].cpu().item(): proposal_pred
                        }
                    )
            # not 'ori_size'
            orig_target_sizes = torch.stack([t['size'] for t in targets], dim=0)
            # selected_proposals = proposal_processor.top_proposals(proposals, target_sizes=orig_target_sizes)
            selected_proposals = proposal_processor(proposals, target_sizes=orig_target_sizes)

            # targets = get_targets_from_proposals(selected_proposals, targets)
            targets = update_targets_with_proposals(selected_proposals, targets)

        model_kwargs = dict()
        # For debugging purpose (e.g., to access fg_gt), we always pass targets
        model_kwargs['sgdt_args'] = (targets,)
        # if is_with_sgdt_layer(args.encoder_layer_config):
        #     model_kwargs['sgdt_args'] = (targets,)
        #     # model_kwargs['sigma'] = (sigma,)
        teacher_outputs = None
        if feature_distillation_teacher_model is not None:
            teacher_model_kwargs = dict()
            # if is_with_sgdt_layer(args.encoder_layer_config):
            teacher_model_kwargs['sgdt_args'] = (targets, )  # gt_ratio_or_sigma,
            if args.skip_teacher_model_decoder_forward:
                teacher_model_kwargs['skip_teacher_model_decoder_forward'] = args.skip_teacher_model_decoder_forward

            with torch.cuda.amp.autocast(enabled=args.amp):
                if feature_distillation_teacher_model.training:
                    dn_args_ = (targets, args.scalar, args.label_noise_scale, args.box_noise_scale,
                                args.num_patterns)
                else:
                    dn_args_ = args.num_patterns

                if need_tgt_for_training:
                    teacher_outputs, _ = feature_distillation_teacher_model(
                        samples,
                        dn_args=dn_args_,
                        **teacher_model_kwargs,
                    )
                else:
                    teacher_outputs = model(samples)

            # update the input variable
            teacher_encoder_output_list = teacher_outputs['encoder_output_list']

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(teacher_encoder_output_list=teacher_encoder_output_list)
                loss_dict = criterion(outputs, sgdt=teacher_outputs['sgdt'])

                # loss_dict = criterion(outputs, targets,
                        #                       src_key_padding_mask=teacher_outputs['src_key_padding_mask'])

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        # # if 'class_error' in loss_dict_reduced:
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        _cnt += 1
        if args.debug:
            if _cnt % break_iter == 0:  # 15
                print("BREAK!" * 5)
                break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    return resstat


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

            if not args.no_resume_optimizer_lr_schedule:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
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

    def get_feature_distillation_pretrained_models():
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
              --feature_distillation \
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

    feature_distillation_teacher_model = None
    if args.feature_attn_distillation == 'separate_trained_model' or args.two_model_self_attn_or_feat_sharing:
        feature_distillation_teacher_model = get_feature_distillation_pretrained_models()


    # =======================
    if args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir,
                                              wo_class_error=wo_class_error,
                                              args=args,
                                              proposal_model=proposal_model,
                                              feature_distillation_teacher_model=feature_distillation_teacher_model,

                                              # proposal_processor=model.sgdt.proposal_processor
                                              )
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return test_stats

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
            distillation_teacher_model=feature_distillation_teacher_model,
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

        test_stats = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args,
            logger=(logger if args.save_log else None),
            # --------
            wandb=wandb,
            proposal_model=proposal_model,
            feature_distillation_teacher_model=feature_distillation_teacher_model,
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
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)
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
