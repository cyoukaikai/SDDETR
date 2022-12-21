#!/usr/bin/env bash
#source ./utils.sh

export CUDA_VISIBLE_DEVICES=2
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************

master_port=29501  



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6 
exp_str=finetune-regular_6-with-offline-feat-distill-from_sMLP_Fg_KV51AP  # ShareV_out_proj_FFN_attn52_2ap
#===========================

python main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 5 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.5 \
  --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
  --with_teacher_model \
  --skip_teacher_model_decoder_forward \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --save_checkpoint_interval 1 \
  --wandb



  
  #   --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  #   --with_sgdt_attention_loss  \
 # --sgdt_attention_loss_coef 5 \
<<COMMENT
   
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6 
exp_str=finetune-lr0.1-regular_6-with-offline-feat-distill-from_sMLP_Fg_KV51AP  # ShareV_out_proj_FFN_attn52_2ap
#===========================

python main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 5 \
  --drop_lr_now \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.5 \
  --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
  --with_teacher_model \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-DualAttnShareVOutProjFFN_1
exp_str=debug-loss_coef2-ShareV-attn-offline-distill-checkpoint0035_beforedrop-from_ShareV_out_proj_FFN_attn52_2ap
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
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 3 \
  --decoder_layer_config regular_4 \
  --with_sgdt_attention_loss  \
  --sgdt_attention_loss_coef 2 \
  --freeze_attn_online_encoder_distillation \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint0035_beforedrop.pth \
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
encoder_layer_config=regular_4-DualAttnShareVOutProjFFN_1
exp_str=debug-loss_coef0.5-ShareV-feat-offline-distill-from_ShareV_out_proj_FFN_attn52_2ap
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
  --sgdt_transformer_feature_distillation_loss_coef 0.5 \
  --freeze_attn_online_encoder_distillation \
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN_lr0.1_from_epoch35/checkpoint.pth \
  --with_teacher_model \
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --skip_teacher_model_decoder_forward \
  --save_checkpoint_interval 1 \
  --wandb
  

#-m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_attention_learning_no_decoder_from_v_shared_double_attn52_2ap
#===========================

python main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 8 \
   --lr_drop 6 \
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
  --teacher_model_use_pretrained_v_shared_double_attn52_2ap \
  --loss_disable_ignore_keywords attention feature \
  --ignore_detr_loss \
  --freeze_detr_decoder \
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.5 \
  --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
  --wandb   
  
 
 
 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
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
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-DualAttnShareVOutProjFFN_1
exp_str=attn-online-freeze-distill-share_double_head_transformer_ShareV_out_proj_FFN
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 1 \
  --eval
  
  
  
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 2 \
  --eval
  
  
  
 
  
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 3 \
  --eval
  
  
  
  
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 4 \
  --eval
  
  
  
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 5 \
  --eval
  
  
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 6 \
  --eval
  
  
  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 11 \
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
    --no_resume_optimizer_lr_schedule \
      --save_checkpoint_interval 1 \
      --sgdt_attention_loss_coef 500 \
  --auto_resume   \
  --debug_st_attn_sweep_n_attn_heads 6 \
  --eval
  
  
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-DualAttnShareVOutProjFFN_1
exp_str=finetune_attn_not_freeze_backbone_encoder
# attn-online-mutual-distill-share_double_head_transformer_ShareV_out_proj_FFN
#===========================

python main.py -m sgdt_dn_dab_detr \
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
  --pretrain_model_path logs/e6-d6-gt_split_only/share_double_head_transformer_ShareV_out_proj_FFN/checkpoint0010.pth \
 --freeze_attn_online_encoder_distillation  \
 --training_only_distill_student_attn_not_free_backbone  \
 --training_skip_forward_decoder  \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-parallelSTECSGDTShareVNoFreeze_1 
exp_str=parallelSTECSGDTShareVNoFreeze_with_online_distillation_freeze_teacher
exp_str_old=parallelSTECSGDTShareVNoFreeze_with_online_distillation

#===========================

python main.py -m sgdt_dn_dab_detr \
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
  --freeze_online_encoder_distillation \
  --eval_decoder_layer 4 \
  --eval 

  
  
pad_fg_pixel=0
token_scoring_gt_criterion=fg_scale_class_all_fg
out_dir=e6-d6-gt_split_only
exp_str=sgdt_token_fg_bg_classifier_pretrained_detr44AP_5e-4-again
#===========================

python main_token_classifier.py -m sgdt_token_fg_bg_classifier \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
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
  
COMMENT


