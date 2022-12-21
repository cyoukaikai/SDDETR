# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from util.utils import to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

# ========================
import smrc.utils
from models.sgdt.scoring_gt import sgdt_update_sample_target, update_targets_with_proposals
from models.sgdt.sgdt_components import is_with_sgdt_layer
from models.sgdt.sgdt_ import init_proposal_processor


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,
                    wandb=None,  # --------
                    gt_ratio_or_sigma=None,
                    proposal_model=None,
                    teacher_model=None,
                    ):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    if teacher_model is not None:
        teacher_model.train()  # param.requires_grad = False for all parameters even in train mode

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    proposal_processor = init_proposal_processor(args.proposal_scoring)

    # samples, targets = data_loader.batch_sampler.sampler.dataset[11270]

    # img_ids = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header,
                                                    logger=logger):
        # if _cnt < 11270:
        #     _cnt += 1
        #     continue
        # if _cnt == 29:
        #     print(f'well, ')
        # _cnt = 11270
        # samples, targets = data_loader.batch_sampler.sampler.dataset[_cnt]

        # ------------------- tti
        gt_ratio_or_sigma.update(cur_epoch=epoch, cur_iter=_cnt)
        samples, targets = sgdt_update_sample_target(samples, targets, args)
        # -------------------

        # image_ids = [t['image_id'].item() for t in targets]
        # # # img_ids.append([epoch, _cnt] + image_ids)
        # print(f'epoch={epoch}, _cnt = {_cnt}, image_ids = {image_ids}')
        # # continue

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if proposal_model is not None:
            with torch.cuda.amp.autocast(enabled=args.amp):
                if need_tgt_for_training:
                    proposals, _ = proposal_model(samples, dn_args=args.num_patterns,)
                else:
                    proposals = proposal_model(samples)
            # not 'orig_size' but 'size'
            orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            # normalized box -> unnormalized  box (box in the image coordinate system)
            selected_proposals = proposal_processor(proposals, target_sizes=orig_target_sizes)
            # targets = get_targets_from_proposals(selected_proposals, targets)
            targets = update_targets_with_proposals(selected_proposals, targets)

        model_kwargs = dict()
        # For debugging purpose (e.g., to access fg_gt), we always pass targets, gt_ratio_or_sigma
        model_kwargs['sgdt_args'] = (targets, gt_ratio_or_sigma,)
        # if is_with_sgdt_layer(args.encoder_layer_config):
        #     model_kwargs['sgdt_args'] = (targets, gt_ratio_or_sigma,)
        if args.training_skip_forward_decoder:
            model_kwargs['training_skip_forward_decoder'] = args.training_skip_forward_decoder

        teacher_outputs = None
        if teacher_model is not None:
            teacher_model_kwargs = dict()
            # if is_with_sgdt_layer(args.encoder_layer_config):
            teacher_model_kwargs['sgdt_args'] = (targets, gt_ratio_or_sigma,)
            if args.skip_teacher_model_decoder_forward:
                teacher_model_kwargs['training_skip_forward_decoder'] = args.skip_teacher_model_decoder_forward

            with torch.cuda.amp.autocast(enabled=args.amp):
                if teacher_model.training:
                    dn_args_ = (targets, args.scalar, args.label_noise_scale, args.box_noise_scale,
                                args.num_patterns)
                else:
                    dn_args_ = args.num_patterns

                if need_tgt_for_training:
                    teacher_outputs, _ = teacher_model(
                        samples,
                        dn_args=dn_args_,
                        **teacher_model_kwargs,
                    )
                else:
                    teacher_outputs = teacher_model(samples)

            # update the input variable
            # teacher_encoder_output_list = teacher_outputs['encoder_output_list']
            # model_kwargs['teacher_encoder_output_list'] = teacher_encoder_output_list
            model_kwargs['teacher_encoder_decoder_out_dict'] = teacher_outputs['encoder_decoder_out_dict']
            # print(f'teacher_encoder_output_list={teacher_encoder_output_list}')

        if args.modelname.lower().startswith('dn_'):
            model_kwargs = dict()

        if not args.token_classifier:  # Normal detection
            with torch.cuda.amp.autocast(enabled=args.amp):
                if need_tgt_for_training:
                    outputs, mask_dict = model(samples,
                                               dn_args=(targets, args.scalar, args.label_noise_scale,
                                                        args.box_noise_scale, args.num_patterns),
                                               **model_kwargs,
                                               )
                    loss_dict = criterion(outputs, targets, mask_dict)

                else:
                    outputs = model(samples)
                    loss_dict = criterion(outputs, targets)

        else:
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(  # TODO: to update the code here
                    teacher_encoder_output_list=model_kwargs['teacher_encoder_output_list']
                )
                loss_dict = criterion(outputs, sgdt=teacher_outputs['sgdt'])

                # loss_dict = criterion(outputs, targets,
                #                       src_key_padding_mask=teacher_outputs['src_key_padding_mask'])

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # metric_logger.update(loss=loss_value)
        if wandb is not None:
            # step = epoch
            tags = dict(total_loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            tags.update(dict(cur_epoch=float(epoch)))
            wandb.log(tags, commit=True)  # step=self.get_iter(runner),
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:  # 15
                print("BREAK!" * 5)
                break

    # smrc.utils.save_1d_list_to_file(file_path=f'epoch{epoch}_img_pairs.txt', list_to_save=img_ids)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,
             wo_class_error=False, args=None, logger=None,
             wandb=None,
             proposal_model=None,
             teacher_model=None,
             eval_decoder_layer=-1, epoch: int = None,
             ):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    if teacher_model is not None:
        teacher_model.eval()

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

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    output_fg_attn = []
    output_fg_attn_map = {}
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
        if teacher_model is not None:
            teacher_model_kwargs = dict()
            # if is_with_sgdt_layer(args.encoder_layer_config):
            teacher_model_kwargs['sgdt_args'] = (targets, )  # gt_ratio_or_sigma,
            if args.skip_teacher_model_decoder_forward:
                teacher_model_kwargs['training_skip_forward_decoder'] = args.skip_teacher_model_decoder_forward

            with torch.cuda.amp.autocast(enabled=args.amp):
                if teacher_model.training:
                    dn_args_ = (targets, args.scalar, args.label_noise_scale, args.box_noise_scale,
                                args.num_patterns)
                else:
                    dn_args_ = args.num_patterns

                if need_tgt_for_training:
                    teacher_outputs, _ = teacher_model(
                        samples,
                        dn_args=dn_args_,
                        **teacher_model_kwargs,
                    )
                else:
                    teacher_outputs = teacher_model(samples)

            # update the input variable
            # teacher_encoder_output_list = teacher_outputs['encoder_output_list']
            # model_kwargs['teacher_encoder_output_list'] = teacher_encoder_output_list
            model_kwargs['teacher_encoder_decoder_out_dict'] = teacher_outputs['encoder_decoder_out_dict']
            # print(f'teacher_encoder_output_list={teacher_encoder_output_list}')

        if args.modelname.lower().startswith('dn_'):
            model_kwargs = dict()

        if not args.token_classifier:
            with torch.cuda.amp.autocast(enabled=args.amp):
                if need_tgt_for_training:
                    outputs, _ = model(samples, dn_args=args.num_patterns,
                                       **model_kwargs,
                                       # sgdt_args=(targets,),
                                       # proposal_processor=proposal_processor if args.use_decoder_proposal else None,
                                       )
                else:
                    outputs = model(samples)
                # outputs = model(samples)

                loss_dict = criterion(outputs, targets)

        else:
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(teacher_encoder_output_list=teacher_outputs['encoder_output_list'])
                loss_dict = criterion(outputs, sgdt=teacher_outputs['sgdt'])

                # loss_dict = criterion(outputs, targets,
                #                       src_key_padding_mask=teacher_outputs['src_key_padding_mask'])

        # weight_dict = criterion.weight_dict
        # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        #
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # # if 'class_error' in loss_dict_reduced:
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if 'fg_attn' in loss_dict:
            # student_fg_attn, teacher_fg_attn = loss_dict['fg_attn']
            # for target, stu_attn, teacher_attn in zip(targets, student_fg_attn, teacher_fg_attn):
            #     k = target['image_id'].cpu().item()
            #     output_fg_attn.append([k, stu_attn.cpu().item(), teacher_attn.cpu().item()])

            fg_attn = loss_dict['fg_attn']
            for target, attn in zip(targets, fg_attn):
                k = target['image_id'].cpu().item()
                output_fg_attn.append([k] + [x.cpu().item() for x in attn])

        if 'attn_map' in loss_dict:
            fg_attn = loss_dict['attn_map']
            for target, attn_map in zip(targets, fg_attn):
                image_id = target['image_id'].cpu().item()
                # output_fg_attn_map[k] = attn_map.cpu()
                fg_attn_map = attn_map.cpu()
                cur_epoch = f'' if epoch is None else f'{epoch}'
                savepath = os.path.join(args.output_dir, f'output_fg_attn_epoch{cur_epoch}/{image_id}.pkl')
                smrc.utils.generate_dir_for_file_if_not_exist(savepath)
                torch.save(fg_attn_map, savepath)

                # smrc.utils.save_multi_dimension_list_to_file(
                #     filename=os.path.join(args.output_dir, 'output_fg_attn.txt'),
                #     delimiter=',',
                #     list_to_save=output_fg_attn,
                # )
        # for the final process, it is 'orig_size' as the original code.
        if eval_decoder_layer is not None and eval_decoder_layer != -1 and \
                eval_decoder_layer != len(outputs['aux_outputs']):  #
            assert isinstance(eval_decoder_layer, int) and eval_decoder_layer < len(outputs['aux_outputs'])
            # change the evaluation layer of the decoder
            # out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            # [{'pred_logits': a, 'pred_boxes': b}
            outputs['pred_logits'] = outputs['aux_outputs'][eval_decoder_layer]['pred_logits']
            outputs['pred_boxes'] = outputs['aux_outputs'][eval_decoder_layer]['pred_boxes']

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # import ipdb; ipdb.set_trace()

        res_cpu = {}
        for target, output in zip(targets, results):
            k = target['image_id'].item()
            v = {kk: vv.cpu() for kk, vv in output.items()}
            v["size"] = list(target["size"].cpu().numpy())
            # v["size"] = target["size"]
            v['padded_img_size'] = list(target['padded_img_size'].cpu().numpy())
            v['padded_size'] = list((target['padded_img_size'] - target["size"]).cpu().numpy())
            v['padded_area'] = (target['padded_img_size'].prod() - target["size"].prod()).cpu().item()
            res_cpu[k] = v

        # for k, v in res.items():
        #     res_cpu[k] = {kk: vv.cpu() for kk, vv in v.items()}
        # {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in res.items()}
        if 'res' not in output_state_dict:
            output_state_dict['res'] = res_cpu
        else:
            output_state_dict['res'].update(res_cpu)

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            """
            saving results of eval.
            """
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']
            # import ipdb; ipdb.set_trace()

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct

                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

                # ------------
                # image_id = tgt['image_id'].item()
                # img_info = dict(image_id=image_id, img_path=f"{image_id:012d}.png")
                # if 'img_info' not in output_state_dict:
                #     output_state_dict['img_info'] = []

                # import ipdb;ipdb.set_trace()
                # output_state_dict['img_info'].append(img_info)

        _cnt += 1
        if args.debug:
            if _cnt % break_iter == 0:  # 15
                print("BREAK!" * 5)
                break

    if len(output_fg_attn) > 0:
        import os.path as osp
        smrc.utils.save_multi_dimension_list_to_file(
            filename=osp.join(args.output_dir, 'output_fg_attn.txt'),
            delimiter=',',
            list_to_save=output_fg_attn,
        )


    # save the proposal
    if proposal_model is not None and proposal_saved_file is not None and proposals_loaded is None:
        smrc.utils.generate_pkl_file(pkl_file_name=proposal_saved_file, data=proposal_to_save)

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()

            # -------------- TTI
            if wandb is not None:
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
                tags = {k: v for k, v in zip(metric_name, stats['coco_eval_bbox'])}
                wandb.log(tags, commit=True)  # step=self.get_iter(runner),
            # ------------------------------

        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    # import ipdb; ipdb.set_trace()
    if args.save_coco_evaluator_prefix is not None:
        coco_evaluator_prefix = 'coco_evaluator'
        if len(args.save_coco_evaluator_prefix) > 0:
            coco_evaluator_prefix = args.save_coco_evaluator_prefix

        savepath = os.path.join(args.output_dir, f'{coco_evaluator_prefix}.pkl')
        print("Saving coco_evaluator to {}".format(savepath))
        torch.save(coco_evaluator, savepath)

    return stats, coco_evaluator
