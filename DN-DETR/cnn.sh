#!/usr/bin/env bash
#source ./utils.sh
 
 
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export USE_GLOG=ON
export TORCH_DISTRIBUTED_DEBUG=INFO # or DETAIL
GPU_NUM=2
#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************
# To run tomorrow
master_port=29501




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-DualAttnShareVOutProjFFN_1
exp_str=AblationStudy_MarkFg1Bg0_QK_regular_5-DualAttnShareVOutProjFFN_1
#===========================
python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
 main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --eval_decoder_layer 5 \
  --decoder_layer_config regular_6 \
  --transformer_type share_double_head_transformer \
  --save_checkpoint_interval 5 \
  --token_masking MarkFg1Bg0  \
  --token_masking_loc  QK \
  --auto_resume \
  --num_workers 6 \
  --wandb
  
  
./cnn.sh
  
  
#python tti/read_coco_results_tool.py -i logs/$out_dir/
   

   #  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN_lr0.1_from_epoch35/checkpoint.pth \
#--drop_lr_now \
#   --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint0035_beforedrop.pth \
# next step:
# do not share w_q; use sgdt not fg_bg_mask for update K.

<<COMMENT

out_dir=regular6
encoder_layer_config=regular_6
exp_str=dn_dab_deformable_detr_lr0.5_x2gpus
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m dn_dab_deformable_detr --output_dir logs/$out_dir/$exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6
  --eval_decoder_layer 5 \
  --encoder_layer_config $encoder_layer_config \
  --decoder_layer_config regular_6 \
  --auto_resume \
  --num_workers 6 \
  --save_checkpoint_interval 2 

out_dir=regular6
exp_str=dn_dab_detr_R50_DC5_lr0.5_x2gpus
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m dn_dab_detr --output_dir logs/$out_dir/$exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --dilation 
#===========================

  

  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=share_double_head_transformer_ShareV_out_proj_FFN_E6D6
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 60 \
  --lr_drop 100 \
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
  --eval_decoder_layer 5 \
  --decoder_layer_config regular_6 \
  --auto_resume \
  --num_workers 6 \
  --save_checkpoint_interval 2 \
   --wandb
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-DualAttnShareV_1
exp_str=confirm_batch2x2_sMLP_K_regular_5-DualAttnShareV_1
#===========================

python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
 main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 5 \
  --decoder_layer_config regular_6 \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 5 \
  --token_masking sMLP  \
  --token_masking_loc  K \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=share_double_head_transformer_ShareV_out_proj_FFN_E6D6
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
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
  --eval_decoder_layer 5 \
  --decoder_layer_config regular_6 \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  
  
  
  
  --drop_lr_now \
  
  
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-offline-distill-from_ShareV_out_proj_FFN_attn52_6ap-third-run-lr-lr0.1-two-teachers
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 10 \
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
  --freeze_attn_online_encoder_distillation \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_6ap_lr1_not_converged \
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 1 \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-offline-distill-from_ShareV_out_proj_FFN_attn52_6ap-second-run
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 9 \
  --lr_drop 6 \
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
  --freeze_attn_online_encoder_distillation \
  --auto_resume \
    --no_resume_optimizer_lr_schedule \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_6ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
    --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 0.01 \
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-learning-from_ShareV_out_proj_FFN_attn52_6ap
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
  --coco_path coco \
  --use_dn \
  --lr 5e-6 \
  --lr_backbone 5e-7 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 5 \
  --freeze_attn_online_encoder_distillation \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_6ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --cls_loss_coef 0.1 \
  --dice_loss_coef 0.1 \
  --bbox_loss_coef 0.5 \
  --giou_loss_coef 0.2 \
  --wandb
  
  
 --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint.pth \
  
#==================================
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
  --lr_drop 500 \
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
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5
exp_str=R50_lr0.5_x2gpus_e5_d4_50epoch
# initialize the decoder weights
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 500 \
  --drop_lr_now \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --decoder_layer_config regular_4 \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5
exp_str=R50_lr0.5_x2gpus_e5_d4_50epoch
# initialize the decoder weights
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 36 \
  --lr_drop 36 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --decoder_layer_config regular_4 \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5
exp_str=R50_lr0.5_x2gpus_e5_d4_50epoch
# initialize the decoder weights
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 40 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
    --decoder_layer_config regular_4 \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 5 \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-and-feat-offline-distill-from_ShareV_out_proj_FFN_attn52_2ap
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 8 \
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
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 5 \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.5 \
  --freeze_attn_online_encoder_distillation \
   --auto_resume \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-offline-distill-checkpoint0035_beforedrop-from_ShareV_out_proj_FFN_attn52_2ap
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 10 \
  --lr_drop 100 \
  --coco_path coco \
  --use_dn \
  --lr 5e-6 \
  --lr_backbone 5e-7 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 5 \
  --freeze_attn_online_encoder_distillation \
   --auto_resume \
   --no_resume_optimizer_lr_schedule \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-offline-distill-checkpoint0035_beforedrop-from_ShareV_out_proj_FFN_attn52_2ap
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 10 \
  --lr_drop 100 \
  --coco_path coco \
  --use_dn \
  --lr 5e-6 \
  --lr_backbone 5e-7 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 5 \
  --freeze_attn_online_encoder_distillation \
   --auto_resume \
   --no_resume_optimizer_lr_schedule \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
]]
  # --auto_resume \
  #   --attn_distillation_teacher_with_grad \
  #   --feature_distillation_teacher_feat_with_grad \
  #  --feature_distillation_teacher_feat_with_grad \
  # --auto_resume \
  #  --no_resume_optimizer_lr_schedule \
  #   --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  # --token_scoring_loss_criterion $token_scoring_loss_criterion  \
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-offline-distill-from_ShareV_out_proj_FFN_attn52_2ap
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 8 \
  --lr_drop 5 \
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
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 5 \
  --freeze_attn_online_encoder_distillation \
  --auto_resume \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
    --save_checkpoint_interval 1 \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=ShareV-attn-offline-distill-from_ShareV_out_proj_FFN_attn52_2ap
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 5 \
  --freeze_attn_online_encoder_distillation \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN_lr0.1_from_epoch35/checkpoint.pth \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
    --save_checkpoint_interval 1 \
  --wandb
  # --auto_resume \
  #   --attn_distillation_teacher_with_grad \
  #   --feature_distillation_teacher_feat_with_grad \
  #  --feature_distillation_teacher_feat_with_grad \
  # --auto_resume \
  #  --no_resume_optimizer_lr_schedule \
  #   --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  # --token_scoring_loss_criterion $token_scoring_loss_criterion  \
python tti/read_coco_results_tool.py -i logs/$out_dir/
   
   
   
   
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6  
exp_str=offline_self_attn_distill2_from_teacher_model_use_pretrained_v_shared_double_attn52_2ap

# regular_5-sgdtSharedAttn_1 
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --decoder_layer_config regular_6 \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --auto_resume \
  --with_sgdt_attention_loss \
  --sgdt_attention_loss_coef 1 \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  --wandb
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-sgdtSharedAttn_1  
exp_str=check_encoder_self_attn_improved_from_teacher_model_use_pretrained_v_shared_double_attn52_2ap

# regular_5-sgdtSharedAttn_1 
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --decoder_layer_config regular_4 \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 5 \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_distillation-loss-coef20_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --auto_resume \
  --wandb   
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_distillation_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --sgdt_attention_loss_coef 50 \
  --auto_resume \
  --wandb   
  
 
## 0,1,2,3,4,5 epochs done

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_and_feature_distillation_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --sgdt_attention_loss_coef 50 \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 1 \
  --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
  --wandb   
  




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_and_feature_distillation_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.1 \
  --auto_resume \
  --wandb   
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_distillation_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --auto_resume \
  --wandb   
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_distillation_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --auto_resume \
  --wandb   
  
  
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_attention_learning_no_decoder_from_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
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
  --sgdt_attention_loss_coef 5 \
  --training_skip_forward_decoder \
  --loss_disable_ignore_keywords attention feature \
  --ignore_detr_loss \
  --freeze_detr_decoder \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0 \
  --wandb   
  
  
  
  #--auto_resume \
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_distillation_t_sMLP_Fg_KV51AP_from44.7model
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
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
  --sgdt_attention_loss_coef 5 \
  --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
  --wandb   
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_independent_attention_distillation_t_sMLP_Fg_KV51AP
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 23 \
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
  --sgdt_attention_loss_coef 10 \
  --auto_resume \
  --wandb   
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=attn-transformer_feature_online-freeze-teacher-distill-ShareV_out_proj_FFN_from_epoch23
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_attention_loss_coef 5 \
  --sgdt_transformer_feature_distillation_loss_coef 1 \
  --freeze_attn_online_encoder_distillation \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint0023.pth \
  --wandb
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=attn-transformer_feature_online-non-mutual-distill-ShareV_out_proj_FFN_from_epoch21
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --attn_distillation_teacher_with_grad \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 1 \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint0021.pth \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=attn-transformer_feature_online-mutual-distill-ShareV_out_proj_FFN_from_epoch21
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --attn_distillation_teacher_with_grad \
  --with_sgdt_attention_loss  \
  --with_sgdt_transformer_feature_distill_loss \
  --feature_distillation_teacher_feat_with_grad \
  --sgdt_transformer_feature_distillation_loss_coef 5 \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint0021.pth \
  --wandb
  
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
  --epochs 24 \
  --lr_drop 22 \
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
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=attn-online-mutual-distill-share_double_head_transformer_ShareV_out_proj_FFN
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 22 \
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
  --attn_distillation_teacher_with_grad \
  --with_sgdt_attention_loss  \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --wandb
  


pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=finetune_attn_top100_not_freeze_backbone_only_regular_4-parallelSTECSGDTShareVOutProjFFN_1
# attn-online-mutual-distill-share_double_head_transformer_ShareV_out_proj_FFN
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
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
  --with_sgdt_attention_loss  \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint.pth \
 --freeze_attn_online_encoder_distillation  \
 --training_only_distill_student_attn_not_free_backbone  \
 --training_skip_forward_decoder  \
 --attention_loss_top_100_token \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVOutProjFFN_1  
exp_str=finetune_attn_only_regular_4-parallelSTECSGDTShareVOutProjFFN_1
# attn-online-mutual-distill-share_double_head_transformer_ShareV_out_proj_FFN
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint.pth \
 --freeze_attn_online_encoder_distillation  \
 --training_only_distill_student_attn  \
 --training_skip_forward_decoder  \
  --wandb
  
  




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1  
exp_str=share_double_head_transformer_ShareV_and_decoder_other_independent
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 5 \
  --decoder_layer_config regular_6 \
  --auto_resume \
  --wandb
  
  
  
    --resume logs/$out_dir/$exp_str/checkpoint0009.pth \
  --pretrain_model_path logs/$out_dir/$exp_str_old/checkpoint0002.pth \
  



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1 
exp_str=online_encoder_mutual_distillation_from_epoch0015pth
exp_str_old=parallelSTECSGDTShareVNoFreeze
#===========================


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --online_self_distill_transformer \
  --attn_distillation_teacher_with_grad \
  --save_checkpoint_interval 1 \
  --with_sgdt_attention_loss  \
  --eval_decoder_layer 4 \
  --pretrain_model_path logs/$out_dir/$exp_str_old/checkpoint0015.pth \
  --auto_resume \
  --wandb



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1 
exp_str=parallelSTECSGDTShareVNoFreeze_with_online_distillation_freeze_teacher_and_student_V
exp_str_old=parallelSTECSGDTShareVNoFreeze_with_online_distillation_freeze_teacher_and_student_V

#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
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
  --online_self_distill_transformer \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 1 \
  --with_sgdt_attention_loss  \
  --freeze_attn_online_encoder_distillation \
  --eval_decoder_layer 4 \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1 
exp_str=parallelSTECSGDTShareVNoFreeze_with_online_distillation_freeze_teacher_and_student_V
exp_str_old=parallelSTECSGDTShareVNoFreeze_with_online_distillation

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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --online_self_distill_transformer \
  --pretrain_model_path logs/$out_dir/$exp_str_old/checkpoint.pth \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 1 \
  --with_sgdt_attention_loss  \
  --freeze_attn_online_encoder_distillation \
  --eval_decoder_layer 4 \
  --wandb
  
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1 
exp_str=parallelSTECSGDTShareVNoFreeze_with_online_distillation
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 22 \
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
  --online_self_distill_transformer \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  --with_sgdt_attention_loss  \
  --eval_decoder_layer 4 \
  --wandb


  
  
  
## 0,1,2,3,4,5 epochs done

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1 
exp_str=parallelSTECSGDTShareVNoFreeze
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --online_self_distill_transformer \
  --auto_resume \
  --wandb
  
  
  
## 0,1,2 epochs done

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdt+k_1 
exp_str=sgdt_transformer_UpdateK
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --align_encoder_decoder_layers_num 1 \
  --auto_resume \
  --wandb
  
  
  


  --resume logs/$out_dir/$exp_str/checkpoint0010.pth \


pad_fg_pixel=0
token_scoring_gt_criterion=significance_value
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-onlineDistilEncoderOnlyShareEverything_1 
exp_str=online_self_distillation_encoder_only_gt_FgBG_K
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --online_self_distill_transformer \
  --auto_resume \
  --wandb 
  
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtSharedAttn_1 
exp_str=check_encoder_self_attn_improved
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 26 \
  --lr_drop 24 \
  --drop_lr_now \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --two_model_self_attn_or_feat_sharing \
  --skip_teacher_model_decoder_forward \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --wandb 
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtSharedAttn_1 
exp_str=check_encoder_self_attn_improved_teacher_model_use_pretrained_detr44AP
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --two_model_self_attn_or_feat_sharing \
  --teacher_model_use_pretrained_detr44AP \
  --skip_teacher_model_decoder_forward \
  --auto_resume \
  --wandb 
 
 
   python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtSharedAttn_1 
exp_str=check_encoder_self_attn_improved
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --two_model_self_attn_sharing \
  --skip_teacher_model_decoder_forward \
  --auto_resume \
  --wandb 


pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=mark_hypethese_marking_encoder_feature_by_fg1_bg0-$encoder_layer_config
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --marking_encoder_feature_by_fg1_bg0 \
  --auto_resume \
  --wandb 
  
  
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_bg
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdt_1
exp_str=mark_hypethese_e6-d6-V-gt_split_only-all-bg-regular_5-sgdt+v_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 3 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt+v_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --auto_resume \
  --wandb 
  
  
 python tti/read_coco_results_tool.py -i logs/$out_dir/
 
  
  
  
  
  
   
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=finetune_recover_accuracy-FFN_out-gt_split_only-regular_5-sgdtv1_1-e6-d6

exp_str_pretrained=gt_split_only-regular_5-sgdtv1_1-e6-d6
# initialize the decoder weights
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
  --auto_resume \
  --freeze_transformer_sgdt_encoder_layer_ffn_out \
  --wandb
  
  python tti/read_coco_results_tool.py -i logs/$out_dir/

#   --pretrain_model_path logs/$out_dir/$exp_str_pretrained/checkpoint.pth \




 # ############################################# 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=finetune_recover_accuracy-MHA_out-gt_split_only-regular_5-sgdtv1_1-e6-d6

exp_str_pretrained=gt_split_only-regular_5-sgdtv1_1-e6-d6
# initialize the decoder weights
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
  --pretrain_model_path logs/$out_dir/$exp_str_pretrained/checkpoint.pth \
  --freeze_transformer_sgdt_encoder_layer_MHA_out \
  --wandb
  
  
   

  python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  
  
  ===================-------
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=finetune_recover_accuracy-FFN_out-gt_split_only-regular_5-sgdtv1_1-e6-d6

exp_str_pretrained=gt_split_only-regular_5-sgdtv1_1-e6-d6
# initialize the decoder weights
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
  --pretrain_model_path logs/$out_dir/$exp_str_pretrained/checkpoint.pth \
  --freeze_transformer_sgdt_encoder_layer_ffn_out \
  --wandb
  
  python tti/read_coco_results_tool.py -i logs/$out_dir/




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=gt_split_only-regular_5-sgdtv1_1-e6-d6
#===========================
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 25 \
  --lr_drop 22 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --drop_lr_now \
  --encoder_layer_config regular_5-sgdtv1_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --auto_resume \
  --wandb
  
  
  
  python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  # eval with larger batch size.



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=gt_split_only-regular_5-sgdtv1_1-e6-d6
#===========================
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 22 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdtv1_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --auto_resume \
  --wandb
  
  
  
  pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=gt_split_only-regular_5-sgdtv1_1-e6-d6
#===========================
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 22 \
  --start_epoch 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdtv1_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --pretrain_model_path logs/$out_dir/$exp_str/checkpoint0010.pth \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp
out_dir=regular_5-sgdt_1
exp_str=e6-d6-gt_only_exp-split-and-remove-regular_5-sgdtv0_1
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
  --encoder_layer_config regular_5-sgdtv0_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --auto_resume \
  --wandb 
  
  
 # I forgot to set feature_distillation but actually feature_distillation is set in the code by default.
  
python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_6
exp_str=f_distil_parallel-small-loss-wo_student_feat_conv-separate_trained_model-gt_split_only-regular_5-sgdtv1_1-align1
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
  --encoder_layer_config regular_6 \
  --auto_resume \
  --feature_distillation separate_trained_model \
  --save_checkpoint_interval 4 \
  --wandb

  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=token_num_no_limit-aligh-sgdtv1_1-debug_split_1c
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
  --eval \
  --auto_resume \
  --attention_map_evaluation
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_6
exp_str=f_distil_parallel-with_student_feat_conv-separate_trained_model-gt_split_only-regular_5-sgdtv1_1-align1
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
  --encoder_layer_config regular_6 \
  --auto_resume \
  --feature_distillation separate_trained_model \
  --save_checkpoint_interval 4 \
  --wandb
  
  

pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=attention_distillation-gt_only_exp-no_bg_token_remove
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
  --with_sgdt_attention_loss \
  --auto_resume \
  --wandb



pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=attention_distillation-gt_only_exp-no_bg_token_remove
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
  --with_sgdt_attention_loss \
  --auto_resume \
  --wandb
  
  
 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_6
exp_str=feature-distillation-separate_trained_model-teach_split1c_version-gt_split_only-regular_5-sgdtv1_1-align1
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
  --encoder_layer_config regular_6 \
  --auto_resume \
  --feature_distillation separate_trained_model \
  --save_checkpoint_interval 4 \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=token_num_no_limit-aligh-sgdtv1_1-debug_split_1c
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
  --wandb
 
  python tti/read_coco_results_tool.py -i logs/$out_dir/
  
    python tti/read_coco_results_tool.py -i logs/regular_5-sgdtv1_1
    
    
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_6
exp_str=feature-distillation-separate_trained_model-teach_split1c_version-gt_split_only-regular_5-sgdtv1_1-align1
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
  --encoder_layer_config regular_6 \
  --auto_resume \
  --feature_distillation separate_trained_model \
  --save_checkpoint_interval 1 \
  --wandb



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=feature-distillation-cascade-new-split1c_version-gt_split_only-aligh-sgdtv1_1
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
  --feature_distillation cascade \
  --save_checkpoint_interval 1 \
  --wandb
  
  
  
    --eval \
   --eval_decoder_layer 4 \

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp
out_dir=regular_5-sgdtv1_1
exp_str=feature-distillation-new-split1c_version-gt_remove_and_split-aligh-sgdtv1_1
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
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp
out_dir=regular_5-sgdtv1_1
exp_str=feature-distillation-new-split1c_version-gt_remove_and_split-aligh-sgdtv1_1
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
 
 
 
 
 
  --eval \
     --eval_decoder_layer 4 \
     
    --eval_decoder_layer 5 \
  
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_split
out_dir=regular_5-sgdtv1_1
exp_str=split1c_version-gt_remove_only-aligh-sgdtv1_1
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
  --wandb

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp
out_dir=regular_5-sgdtv1_1
exp_str=feature-distillation-split1c_version-gt_remove_and_split-aligh-sgdtv1_1
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
  --eval_decoder_layer 4 \
  --feature_distillation \
  --wandb



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-token_num_no_limit
out_dir=regular_5-sgdtv1_1
exp_str=feature-distillation_remove_and_split-aligh-sgdtv1_1-debug_split_1c
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
  --eval_decoder_layer 4 \
  --feature_distillation \
  --wandb
 

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_split_1c
out_dir=regular_5-sgdtv1_1
exp_str=feature-distillation-aligh-sgdtv1_1-debug_split_1c
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
  --eval_decoder_layer 4 \
  --feature_distillation \
  --wandb
 
  --save_checkpoint_interval 8 \
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_split_1c
out_dir=regular6
exp_str=R50_lr0.5_x2gpus_finetune_compare_encoder_feature

  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr --output_dir logs/$out_dir/$exp_str \
  --exp_str  "encoder-feature-better"-$exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_6 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
    --finetune_ignore decoder \
  --pretrain_model_path logs/$out_dir/$exp_str/checkpoint.pth \
  --wandb
 
 
 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_4-sgdtv1_2
exp_str=again-transformer-aligh-sgdtv1_2-use_decoder_proposal_with_gt_decay_small_to_large
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
  --encoder_layer_config regular_4-sgdtv1_2 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --use_decoder_proposal  \
  --align_encoder_decoder_layers_num 2 \
  --proposal_scoring min_split_score0.3-nms_thd0.6 \
  --proposal_token_scoring_gt_criterion significance_value \
  --gt_decay_criterion start_epoch8-end_epoch11 \
  --auto_resume \
  --save_checkpoint_interval 8 \
  --wandb

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_4-sgdtv1_2
exp_str=transformer-aligh-sgdtv1_2-use_decoder_proposal_with_gt_decay
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
  --encoder_layer_config regular_4-sgdtv1_2 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --use_decoder_proposal  \
  --align_encoder_decoder_layers_num 2 \
  --proposal_scoring min_split_score0.3-nms_thd0.6 \
  --proposal_token_scoring_gt_criterion confidence_value \
  --gt_decay_criterion start_epoch8-end_epoch11 \
  --auto_resume \
  --save_checkpoint_interval 8 \

  --wandb
  
  
  --exp_str  $exp_str \


 out_dir=regular6
exp_str=dn_dab_deformable_detr_lr0.5_x2gpus
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m dn_dab_deformable_detr --output_dir logs/$out_dir/$exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6



out_dir=regular6
exp_str=dn_dab_detr_R50_DC5_lr0.5_x2gpus
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m dn_dab_detr --output_dir logs/$out_dir/$exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --dilation 
#===========================




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=finetune-sgdt_transformer-use_decoder_proposal
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --sgdt_transformer \
  --use_decoder_proposal  \
  --proposal_scoring min_split_score0.3-nms_thd0.6 \
  --wandb  \
  --resume logs/$out_dir/$exp_str/checkpoint0002.pth \
  --drop_lr_now 
 
 
 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=freeze-weights-finetune-sgdt_transformer-use_decoder_proposal
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --sgdt_transformer \
  --pretrain_model_path logs/regular_5-sgdt_1/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear/checkpoint.pth \
  --use_decoder_proposal  \
  --proposal_scoring min_split_score0.3-nms_thd0.6 \
  --freeze_sgdt_transformer_trained_layers \
  --wandb  
  

  

  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-use_proposal
out_dir=regular_5-sgdt_1
exp_str=finetune-split-use_proposal-min_split_score0.3-nms_thd0.6
# min_fg_score0.05-min_split_score0.5-nms_thd0.6-no_split-proposal_remove_gt_filter
#===========================
  
  
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 3 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --decay_sigma \
  --pretrain_model_path logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear/checkpoint.pth \
  --proposal_scoring min_split_score0.3-nms_thd0.6 \
  --wandb

  
  
  
  
    --epochs 12 \
  --lr_drop 11 \
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear
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
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb \
  --sgdt_transformer  \
  --resume logs/$out_dir/$exp_str/checkpoint.pth 
  

 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=topk_token_selection_differentiable-split_2c-with_global_feat-no_bg_token_remove-reclaim_padded_region-best_num_split
out_dir=regular_5-sgdt_1
exp_str=finetune-split-R50-regular_5-sgdt_1-topk_token_selection_differentiable-with_global_feat
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
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --decay_sigma \
  --pretrain_model_path logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear/checkpoint.pth \
  --wandb
  
  python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/split-1c-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0 \
  --exp_str   again-bug-fixed-src-key-mask-split-1c-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --resume logs/$out_dir/split-1c-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0/checkpoint.pth \
  --wandb  
  
  
  
  
  pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=topk_token_selection_differentiable-with_global_feat-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/split-R50-regular_5-sgdt_1-topk_token_selection_differentiable-with_global_feat \
  --exp_str   split-R50-regular_5-sgdt_1-topk_token_selection_differentiable-with_global_feat \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --decay_sigma \
  
  --wandb  
  


  ./exp1.sh
  
  
  #===========================
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-area-max \
  --exp_str   gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-area-max \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb  \
  --resume logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-area-max/checkpoint.pth
  
python tti/read_coco_results_tool.py -i logs/$out_dir/


  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-area-max \
  --exp_str  sgdt_transformer-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-area-max \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb \
  --sgdt_transformer \
  --resume logs/$out_dir/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-area-max/checkpoint.pth
  
  # --resume logs/$out_dir/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16/checkpoint.pth
  
  python tti/read_coco_results_tool.py -i logs/$out_dir/
  # --resume logs/$out_dir/R50/checkpoint.pth


  
  
  pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/sig-value-bug-fixed-interpolation-nearest--R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16\
  --exp_str  sig-value-bug-fixed-interpolation-nearest-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb 
  
  
  pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16\
  --exp_str  sgdt_transformer-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb \
  --sgdt_transformer 
  # --resume logs/$out_dir/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16/checkpoint.pth
  
  python tti/read_coco_results_tool.py -i logs/$out_dir/
  # --resume logs/$out_dir/R50/checkpoint.pth

  ./exp1.sh
  
  
  
  pad_fg_pixel=0
token_scoring_loss_criterion=fg_scale_class_pred_focal_loss
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=pred_token_fg-no_bg_token_remove-reclaim_padded_region-split_only_ambiguous_token-split_sig_thd0.2
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-regular_5-sgdt_1-split_only_ambiguous_token-split_sig_thd0.2-pad0 \
  --exp_str  regular_5-sgdt_1-split_only_ambiguous_token-split_sig_thd0.2-pad0 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb 
  
  
  
  
  pad_fg_pixel=16
token_scoring_loss_criterion=fg_scale_class_pred_focal_loss
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=pred_token_fg-no_bg_token_remove-reclaim_padded_region-split_only_ambiguous_token
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-regular_5-sgdt_1-split_only_ambiguous_token \
  --exp_str  regular_5-sgdt_1-split_only_ambiguous_token \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb 
  
  
  pad_fg_pixel=32
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_split-use_proposal-proposal_remove_gt_filter
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/with-proposal-R50-regular_5-sgdtv1_1-no_split \
  --exp_str  min_fg_score0.05-min_split_score0.5-nms_thd0.6-no_split-proposal_remove_gt_filter \
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
  --proposal_scoring min_fg_score0.05-min_split_score0.5-nms_thd0.6 \
  --wandb 
  
#===========================
pad_fg_pixel=16
token_scoring_loss_criterion=reg_sigmoid_l1_loss
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=pred_significance-reclaim_padded_region-split_sig_thd0.5-filter_false_split-bg_sig_thd0.5-filter_false_remove
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --wandb 
  
  
  
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_conv_pred_fg_only_remove_0.1_debug/R50 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion   conv_pred_fg_only_remove_0.1 \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 
  
  
python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_pred_fg_only_remove_0.1_wo_gf_0.1_pad_fg_pixel_32/R50 \
  --batch_size 2 \
  --epochs 3 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  pred_fg_only_remove_0.1_wo_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 32 \
  --wandb 
  
  
  
  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/5-sgdt_1_v0_with_gt_only_remove/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  v0_with_gt_only_remove  \
  --token_scoring_loss_criterion gt_fg_scale_fake  \
  --token_scoring_gt_criterion significance_value \
  --wandb 
  
  
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_regular_5-sgdt_1_pred_significance_all_fg_w_priority/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --disable_fg_scale_supervision \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  pred_significance_all_fg_w_priority  \
  --token_scoring_loss_criterion reg_sigmoid_l1_loss  \
  --token_scoring_gt_criterion significance_value #\
  #--wandb 
 
 #  # gt_fg_scale_fake
# regular_5-sgdt_1 pred_significance_all_fg_w_priority
# pred_significance_all_fg_w_priority 
# v0_with_gt_only_reclaim_padded_region
# v0_with_gt

#./exp1.sh

    --nproc_per_node=2 --master_port=29500   /disks/cnn1/kaikai/project/DN-DETR/main.py -m sgdt_dn_dab_detr   --output_dir /disks/cnn1/kaikai/project/DN-DETR/logs/sgdt_dn_dab_detr_lr0.5_x2gpus_regular_5-sgdt_1_pred_significance_all_fg_w_priority/R50   --batch_size 2   --epochs 12   --lr_drop 11   --coco_path /disks/cnn1/kaikai/project/DN-DETR/coco   --use_dn   --lr 5e-5 --lr_backbone 5e-6  --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion  v1_selection_differentiable   --token_scoring_loss_criterion fg_scale_class_pred   --token_scoring_gt_criterion fg_scale_class --disable_fg_scale_supervision
    
================== to do 
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_regular_4-sgdtv1_1-sgdt_1/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --disable_fg_scale_supervision \
  --encoder_layer_config regular_4-sgdtv1_1-sgdt_1 \
  --token_scoring_discard_split_criterion  v0_with_gt \
  --wandb 
================== to do 


  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_ec_1l_with_gt_token_label/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 0  \
  --num_encoder_sgdt_layers 1 \
  --disable_fg_scale_supervision \
  --wandb \
  --resume logs/sgdt_dn_dab_detr_lr0.5_x2gpus_ec_1l_with_gt_token_label/R50/checkpoint.pth
  
\
  --encoder_without_pos
  
  
logs/sgdt_token_w_su_focal_loss_after_debug_lr0.5_x2gpus/R50

 \
  --resume logs/sgdt_adapt_with_supervision_dn_dab_detr_lr0.5_x2gpus_debug_done_sgdt_loss_weight0.1/R50/checkpoint.pth
  --wandb 
    \

  
  
  -m sgdt_dn_dab_detr --output_dir logs/dn_dab_detr/tttt  --batch_size 2 --epochs 12 --lr_drop 11 --coco_path coco --use_dn  --lr 2.5e-5 --lr_backbone 2.5e-6  --num_encoder_sgdt_layers 1 --enc_layers 5
  
\
  --sgdt_loss_weight 0.1

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
  
  
    main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_token_wo_su_focal_loss_after_debug_lr0.5_x2gpus/R50 \
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
  --wandb 
  
  
  
  
  ======== supervsion.
    main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_e5l_sgdt1l_pred_score_sumbel_softmax/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 5  \
  --num_encoder_sgdt_layers 1 \
  --disable_fg_scale_supervision \
  --token_scoring_discard_split_criterion  v1_selection_differentiable  \
  --wandb 
COMMENT


