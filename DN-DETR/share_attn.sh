#!/usr/bin/env bash
#source ./utils.sh

#export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export USE_GLOG=ON
export TORCH_DISTRIBUTED_DEBUG=INFO # or DETAIL
GPU_NUM=8
#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************
# To run tomorrow

master_port=29500  



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-DualAttnShareAttnOutProjFFN_1
exp_str=AblationStudy_Second_sMLP_regular_5-DualAttnShareAttnOutProjFFN_1
#===========================

python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
 main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 8 \
  --epochs 50 \
  --lr_drop 40 \
  --coco_path coco \
  --use_dn \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --pad_fg_pixel $pad_fg_pixel \
  --transformer_type share_double_head_transformer \
  --encoder_layer_config $encoder_layer_config \
  --decoder_layer_config regular_6 \
  --token_masking sMLP  \
  --token_masking_loc  V \
  --auto_resume \
  --eval_decoder_layer 5 \
  --save_checkpoint_interval 5 \
  --num_workers 6 \
  --wandb
  
  
  
 ./share_attn.sh
  
<<COMMENT
  --lr 5e-5 \
  --lr_backbone 5e-6 \
COMMENT

