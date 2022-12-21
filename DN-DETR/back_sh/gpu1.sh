#!/usr/bin/env bash
#source ./utils.sh

export CUDA_VISIBLE_DEVICES=1
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-DualAttnShareVOutProjFFN_1
exp_str=debug-loss_coef1-ShareV-feat-offline-distill-from_ShareV_out_proj_FFN_attn52_2ap
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
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 1 \
  --freeze_attn_online_encoder_distillation \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN_lr0.1_from_epoch35/checkpoint.pth \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
 

<<COMMENT

 
pad_fg_pixel=0
token_scoring_gt_criterion=fg_scale_class_all_fg
out_dir=e6-d6-gt_split_only
exp_str=sgdt_token_fg_bg_classifier_sMLP_Fg_KV_lr_5e-4-again
#===========================

python main_token_classifier.py -m sgdt_token_fg_bg_classifier \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 4 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-4 \
  --lr_backbone 5e-5 \
   --token_classifier \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --two_model_self_attn_or_feat_sharing \
  --skip_teacher_model_decoder_forward \
  --wandb 


  
pad_fg_pixel=0
token_scoring_gt_criterion=fg_scale_class_all_fg
out_dir=e6-d6-gt_split_only
exp_str=sgdt_token_fg_bg_classifier_pretrained_detr44AP_5e-4-again
#===========================

python main_token_classifier.py -m sgdt_token_fg_bg_classifier \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 4 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-4 \
  --lr_backbone 5e-5 \
--token_classifier \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --two_model_self_attn_or_feat_sharing \
  --skip_teacher_model_decoder_forward \
  --teacher_model_use_pretrained_detr44AP \
  --wandb 
 
 
 
   python tti/read_coco_results_tool.py -i logs/$out_dir/
 
 
 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
   main.py -m sgdt_dn_dab_detr \
    --coco_path coco \
    --batch_size 2 \
    --use_dn \
    --output_dir logs/train_eval_results \
    --eval \
    --encoder_layer_config regular_6 \
    --resume logs/R50_lr0.5_x2gpus/checkpoint.pth \
    --token_scoring_discard_split_criterion gt_only_exp \
    --token_scoring_loss_criterion reg_sigmoid_l1_loss \
    --token_scoring_gt_criterion significance_value 
    
    #--save_coco_evaluator_prefix coco_evaluator_original_detr
    
    #--debug \
    #--debug_eval_iter 100
    
    
  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_ec_sgdt_only_v1_multi_6l_with_gt_token_label/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 0  \
  --num_encoder_sgdt_layers 6 \
  --encoder_sgdt_layer_version 1 \
  --disable_disable_fg_scale_supervision \
  --wandb 
  
  --enc_layers 6  \
  --wandb \
  --encoder_without_pos
  
  
 --num_encoder_sgdt_layers 0 --enc_layers 1  --debug --encoder_without_pos
 
 
 main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_adapt_with_supervision_dn_dab_detr_lr0.25_x2gpus_debug_done_sgdt_loss_weight1.0/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 2.5e-5 \
  --lr_backbone 2.5e-6 \
  --num_encoder_sgdt_layers 1  \
  --enc_layers 5  \
  --sgdt_loss_weight 1.0 \
  --resume logs/sgdt_adapt_with_supervision_dn_dab_detr_lr0.25_x2gpus_debug_done_sgdt_loss_weight1.0/R50/checkpoint.pth
  
  
  
  
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=$master_port\
  main.py -m dn_dab_detr \
  --output_dir logs/dn_DABDETR-1gpu-lr0.25/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 2.5e-5 \
  --lr_backbone 2.5e-6 \
  --resume logs/dn_DABDETR_lr0.25/R50/checkpoint.pth
  
  --resume logs/sgdt_adapt_with_supervision_dn_dab_detr_lr0.25_x2gpus/R50/checkpoint.pth
2022-7-24
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port\
  main.py -m dn_dab_detr \
  --output_dir logs/dn_DABDETR_lr0.25/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 2.5e-5 \
  --lr_backbone 2.5e-6 \
  --resume logs/dn_DABDETR_lr0.25/R50/checkpoint.pth
2022-8-7
   main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_token_w_su_after_debug_lr0.5_x2gpus/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --num_encoder_sgdt_layers 1  \
  --enc_layers 5  \
  --sgdt_loss_weight 1.0 \
  --disable_fg_scale_supervision \
  --wandb \
  --resume logs/sgdt_token_w_su_after_debug_lr0.5_x2gpus/R50/checkpoint.pth
  
  2022-8-9
  
 main.py -m sgdt_dn_dab_detr \
  --output_dir logs/dn_dab_detr_lr0.5_x2gpus_encoder_without_pos/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 3  \
  --wandb 
  
    2022-8-10
    main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_ec_sgdtv1_1l_with_gt_token_label/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 0  \
  --num_encoder_sgdt_layers 1 \
  --encoder_sgdt_layer_version 1 \
  --disable_fg_scale_supervision \
  --wandb
COMMENT


