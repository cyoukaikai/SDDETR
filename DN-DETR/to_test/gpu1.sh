#!/usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=INFO # or DETAIL
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
log_date=$(date +"%y-%m-%d") # -%M-%S

GPU_NUM=2
#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************
master_port=29501




pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-DualAttnShareV_1
exp_str=kaikai_gcp_batch2x2_MarkFg1Bg0_QK_regular_5-DualAttnShareV_1
#===========================
python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
 main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 4 \
  --epochs 50 \
  --lr_drop 40 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --share_double_head_transformer \
  --eval_decoder_layer 5 \
  --decoder_layer_config regular_6 \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 5 \
  --token_masking MarkFg1Bg0  \
  --token_masking_loc QK \
  --num_workers 12 \
  --wandb
  
./gpu1.sh
 
 
 # --token_scoring_gt_criterion $token_scoring_gt_criterion \
 # --pad_fg_pixel $pad_fg_pixel \

#    # 0.1, 0,1,2,3; 0.5, 4, 5, 6, 7, 8; from 9, 1
 


<<COMMENT
 #  --lr 5e-5 \
  --lr_backbone 5e-6 \
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVNoFreeze_1 
exp_str=share_double_head_transformer_ShareV
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
  --eval_decoder_layer 3 \
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
encoder_layer_config=regular_5
decoder_layer_config=regular_3-STMarkFgBgKShareQVFFN_1
exp_str=decoder_MarkFgBg_CrossAttention-bug-fixed-no-teacher_layer_back_propagate_to_input-E5D4-$decoder_layer_config
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
  --decoder_layer_config $decoder_layer_config  \
  --eval_decoder_layer 3 \
  --online_decoder_self_distill_transformer \
  --auto_resume \
  --wandb
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5
decoder_layer_config=STMarkFgBgKShareQVFFN_4
exp_str=decoder_MarkFgBg_CrossAttention-bug-fixed-E5D4-$decoder_layer_config
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
  --encoder_layer_config $encoder_layer_config \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --decoder_layer_config $decoder_layer_config \
  --dual_attn_transformer \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  --eval_decoder_layer 6 \
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
encoder_layer_config=regular_6
exp_str=decoder_MarkFgBg_CrossAttention-regular_4-STMarkFgBgKShareQVFFN_1
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
  --decoder_layer_config regular_4-STMarkFgBgKShareQVFFN_1  \
  --eval_decoder_layer 4 \
  --online_decoder_self_distill_transformer \
  --auto_resume \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular6
encoder_layer_config=regular_6
exp_str=R50_lr0.5_x2gpus_44epoch
# initialize the decoder weights
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 50 \
  --lr_drop 100 \
  --drop_lr_now \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config $encoder_layer_config \
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 4 \
  --wandb
  
   python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-sgdtSharedAttn_1  
exp_str=check_encoder_self_attn_improved_from_attn_learning_model

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
  --teacher_model_use_pretrained_attn_learning_model \
  --skip_teacher_model_decoder_forward \
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
exp_str=offline_feat_distill_from_teacher_model_use_pretrained_v_shared_double_attn52_2ap

# regular_5-sgdtSharedAttn_1 
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
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
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.1 \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2  \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5 
exp_str=offline_feat_distill_from_teacher_model_use_pretrained_v_shared_double_attn52_2ap

# regular_5-sgdtSharedAttn_1 
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 4 \
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
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.1 \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2  \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5  
exp_str=offline_feat_distill_from_teacher_model_use_pretrained_v_shared_double_attn52_2ap

# regular_5-sgdtSharedAttn_1 
# 0.1, 0,1,2,3; 0.5, 4, 5, 6, 7, 8; from 9, 1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 9 \
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
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 0.5 \
   --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  --wandb


pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5
exp_str=offline_feat_distill_from_teacher_model_use_pretrained_v_shared_double_attn52_2ap

# regular_5-sgdtSharedAttn_1 
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --drop_lr_now \
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
  --with_sgdt_transformer_feature_distill_loss \
  --sgdt_transformer_feature_distillation_loss_coef 1 \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  --wandb
   
   
 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_4-parallelSTECSGDTShareVNoFreeze_1 
exp_str=share_double_head_transformer_ShareV
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
  --auto_resume \
  --no_resume_optimizer_lr_schedule \
  --save_checkpoint_interval 2 \
  --wandb
  
  
  
  ==================================
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
  
  
  
python tti/read_coco_results_tool.py -i logs/$out_dir/

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=ST_attention_learning_no_decoder_from_sMLP_Fg_KV51AP_resume_from44.7detr
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
  --pretrain_model_path logs/checkpoint_optimized_44.7ap.pth \
  --wandb   
  
    

  

  

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
  --auto_resume \
  --wandb


pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=DecoderShareSelfAttnButMarkMemoryQKV
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
  --decoder_layer_config regular_5-regular+Mask_1  \
  --align_encoder_decoder_layers_num 1 \
  --auto_resume \
  --wandb

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=online_decoder_self_distill_transformer-regular_4-STMarkECFFNFeatureFgBgShareV_1
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
  --decoder_layer_config regular_4-STMarkECFFNFeatureFgBgShareV_1  \
  --eval_decoder_layer 4 \
  --online_decoder_self_distill_transformer \
  --auto_resume \
  --wandb
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_3-parallelSTECSGDTShareVNoFreeze_1 
exp_str=online_encoder_mutual_distillation
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
  --online_self_distill_transformer \
  --attn_distillation_teacher_with_grad \
  --save_checkpoint_interval 5 \
  --with_sgdt_attention_loss  \
  --eval_decoder_layer 4 \
  --auto_resume \
  --wandb









pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_6
exp_str=online_decoder_self_distill_transformer-regular_4-STMarkECFFNFeatureFgBgShareQV_1
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
  --decoder_layer_config regular_4-STMarkECFFNFeatureFgBgShareQV_1  \
  --eval_decoder_layer 4 \
  --online_decoder_self_distill_transformer \
  --auto_resume \
  --wandb
  

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtMarkFgBg_1
exp_str=mark_hypethese_marking_encoder_K_by_fg1_bg0-$encoder_layer_config
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
  --marking_encoder_layer_fg1_bg0 K \
  --auto_resume \
  --wandb
  
    

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtMarkFgBg_1
exp_str=mark_hypethese_marking_encoder_V_by_fg1_bg0-$encoder_layer_config
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
  --marking_encoder_layer_fg1_bg0 V \
  --auto_resume \
  --wandb



  #===========================

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
exp_str=e6-d6-gt_split_only-regular_5-sgdt+k_1
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
  --encoder_layer_config regular_5-sgdt+k_1 \
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --auto_resume  \
  --wandb
  
   python tti/read_coco_results_tool.py -i logs/$out_dir/
   
   
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtMarkFgBg_1
exp_str=mark_hypethese_marking_encoder_feature_by_fg1_bg0-$encoder_layer_config
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
  --marking_encoder_layer_fg1_bg0 FFN_Out \
  --eval \
  --auto_resume 

    python tti/read_coco_results_tool.py -i logs/$out_dir/
  
  
  
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
  --epochs 12 \
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

    python tti/read_coco_results_tool.py -i logs/$out_dir/
 
 
 
 

#===========================
 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdtMarkFgBg_1 
exp_str=mark_hypethese_marking_K_by_fg1_bg0


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str-$encoder_layer_config \
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
  --marking_encoder_layer_fg1_bg0 K \
  --auto_resume \
  --wandb 

  
     python tti/read_coco_results_tool.py -i logs/$out_dir/
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  ======================================
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=e6-d6-Q-gt_split_only-regular_5-sgdt_1
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
  --auto_resume \
  --wandb
 
   python tti/read_coco_results_tool.py -i logs/$out_dir/
  



# ############################################# 
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdt+mha+out_1
exp_str=e6-d6-gt_split_only-$encoder_layer_config

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
  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion  \
  --token_scoring_loss_criterion $token_scoring_loss_criterion  \
  --token_scoring_gt_criterion $token_scoring_gt_criterion \
  --pad_fg_pixel $pad_fg_pixel \
  --auto_resume \
  --wandb
 
   python tti/read_coco_results_tool.py -i logs/$out_dir/
   
   
 

 
   
# #############################################  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=e6-d6-gt_split_only
encoder_layer_config=regular_5-sgdt+ffn+out_1
exp_str=e6-d6-gt_split_only-$encoder_layer_config

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
exp_str=gt_split_only-regular_5-sgdtv1_1-e6-d6
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
  --wandb
  
  
  


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port main.py -m sgdt_dn_dab_detr   --coco_path coco   --batch_size 2   --use_dn   --output_dir logs/regular_5-sgdtv1_1/token_num_no_limit-aligh-sgdtv1_1-debug_split_1c --encoder_layer_config regular_5-sgdtv1_1  --token_scoring_discard_split_criterion gt_only_exp-no_bg_token_remove  --token_scoring_loss_criterion gt_fg_scale_fake   --token_scoring_gt_criterion significance_value  --pad_fg_pixel 0  --align_encoder_decoder_layers_num 1  --eval  --pretrain_model_path logs/regular_5-sgdtv1_1/token_num_no_limit-aligh-sgdtv1_1-debug_split_1c/checkpoint.pth  --attention_map_evaluation
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=f_distil_parallel-with_student_feat_conv-gt_split_only-aligh-sgdtv1_1-not-freeze-teacher-feature
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
  --feature_distillation parallel \
  --feature_distillation_teacher_feat_with_grad \
  --eval_decoder_layer 4 \
  --wandb
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove
out_dir=regular_5-sgdtv1_1
exp_str=average_attention_distillation-gt_split_only
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
token_scoring_discard_split_criterion=gt_only_exp
out_dir=regular_5-sgdtv1_1
exp_str=gt_split_and_remove
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
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_split_1c-token_num_no_limit
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
  
  
  --auto_resume \
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_4-sgdtv1_2-gt
exp_str=decay-from-epoch1-transformer-aligh-sgdtv1_2-use_decoder_proposal_with_gt_decay
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
  --gt_decay_criterion start_epoch1-end_epoch11 \
  --save_checkpoint_interval 8 \
  --wandb
  
  
  
out_dir=regular6
exp_str=R50_lr0.5_x2gpus_original

  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m dn_dab_detr --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 24 \
  --lr_drop 20 \
  --start_epoch 12 \
  --coco_path coco \
  --use_dn \
  --lr 5e-6 \
  --lr_backbone 5e-7 \
  --encoder_layer_config regular_6 \
  --pretrain_model_path logs/$out_dir/$exp_str/checkpoint.pth \
  --wandb 
  
  #   --drop_lr_now \
  



out_dir=regular6
exp_str=dn_dab_detr_R50_DC5_lr0.5_x2gpus
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m dn_dab_detr --output_dir logs/$out_dir/$exp_str \
  --exp_str  $exp_str \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --dilation \
  --wandb 

#===========================
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=lr0.1-trained-weights-finetune-sgdt_transformer-use_decoder_proposal
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
  --auto_resume \
  --use_decoder_proposal  \
  --proposal_scoring min_split_score0.3-nms_thd0.6 \
  --lr_sgdt_transformer_trained_layers \
  --wandb  
  
  
  
  
  --pretrain_model_path logs/regular_5-sgdt_1/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear/checkpoint.pth \
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=lr0.1_finetune-sgdt_transformer-use_decoder_proposal
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
  --lr 5e-6 \
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
  --wandb  
  
  
  
  
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdtv1_1
exp_str=regular_5-sgdtv1_1-two-bug-fixed-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0
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
  --wandb   \
  --resume logs/$out_dir/$exp_str/checkpoint.pth 
  
  
pad_fg_pixel=32
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=finetune-split-only-pad0-bilinear-to-pad32
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str \
  --exp_str   $exp_str \
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
  --resume logs/$out_dir/$exp_str/checkpoint0002.pth  \
  --wandb  
  
  
pad_fg_pixel=32
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=finetune-significance_value-split-only-pad0-bilinear-to-fg-all-pad32
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/$exp_str/R50 \
  --exp_str   $exp_str \
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
  --pretrain_model_path logs/regular_5-sgdt_1/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear/checkpoint.pth \
  --wandb  
  
 
  python tti/read_coco_results_tool.py -i logs/$out_dir/$exp_str/
  
  
#===========================
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=topk_token_selection_differentiable-with_global_feat-no_bg_token_remove-reclaim_padded_region-best_num_split
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/split_1c-gt_num_split-R50-regular_5-sgdt_1-topk_token_selection_differentiable-with_global_feat-best_num_split \
  --exp_str   split_1c-best_num_split-R50-regular_5-sgdt_1-topk_token_selection_differentiable-with_global_feat\
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
  
  
 
--nproc_per_node=2 --master_port=29501   main.py -m sgdt_dn_dab_detr   --output_dir logs/ttt  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1   --token_scoring_loss_criterion  gt_fg_scale_fake --token_scoring_gt_criterion significance_value --token_scoring_discard_split_criterion topk_token_selection_differentiable-with_global_feat-no_bg_token_remove-reclaim_padded_region-gt_num_split  --pad_fg_pixel 0 --debug  --decay_sigma  --debug_eval_iter 100
 
#===========================
pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gumbel_softmax_token_selection_differentiable-with_global_feat-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
#===========================


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/split-R50-regular_5-sgdt_1-gumbel_softmax_token_selection_differentiable-with_global_feat \
  --exp_str   split-R50-regular_5-sgdt_1-gumbel_softmax_token_selection_differentiable-with_global_feat \
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
  --sgdt_transformer 
  

  
  
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
  --sgdt_transformer 
  #--resume logs/$out_dir/sgdt_transformer-R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16/checkpoint.pth
  
  
#===========================
pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-num_tokens_to_increase100
out_dir=regular_5-sgdt_1
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-num_tokens_to_increase100\
  --exp_str   gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-num_tokens_to_increase100 \
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
  --resume logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-num_tokens_to_increase100/checkpoint.pth
  
# python tti/read_coco_results_tool.py -i  logs/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-fg_scale_class_small_medium_random
python tti/read_coco_results_tool.py -i logs/$out_dir/
#===========================

pad_fg_pixel=16
token_scoring_loss_criterion=fg_scale_class_pred_focal_loss
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=pred_token_fg_conv-no_bg_token_remove-reclaim_padded_region-split_only_ambiguous_token
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-regular_5-sgdt_1-split_only_ambiguous_token-pred_token_fg_conv \
  --exp_str  regular_5-sgdt_1-split_only_ambiguous_token-pred_token_fg_conv \
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

#===========================
pad_fg_pixel=32  # ---
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg 
token_scoring_discard_split_criterion=gt_only_exp-use_proposal-proposal_split_gt_filter-proposal_remove_gt_filter  #
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-finetune-from-epoch9 \
  --exp_str  finetune_min_fg_score0.05-min_split_score0.5-nms_thd0.6-from-epoch9-with-gt-filtering \
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
  --proposal_scoring min_fg_score0.05-min_split_score0.5-nms_thd0.6 \
  --resume logs/gt_only_exp-32-gt_fg_scale_fake-fg_scale_class_all_fg/R50/back/checkpoint0009.pth \
  --wandb 
  
  
  
pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg  #
token_scoring_discard_split_criterion=gt_only_exp-no_split
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion"-"22-09-04 
===========================

python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
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
  --eval \
  --resume logs/gt_only_exp-no_split-debug_gt_remove_ratio0.5-32-gt_fg_scale_fake-fg_scale_class_all_fg/R50/checkpoint.pth



pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value  #
token_scoring_discard_split_criterion=gt_only_exp-no_split
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion"-"22-09-04 
===========================
python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/regular_5-sgdtv1_1-gt_only_exp-no_split-pad$pad_fg_pixel-R50 \
  --exp_str regular_5-sgdtv1_1-gt_only_exp-no_split-pad16 \
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
  --wandb \
  --resume logs/gt_only_exp-no_split-16-gt_fg_scale_fake-significance_value-22-09-04/regular_5-sgdtv1_1-gt_only_exp-no_split-pad$pad_fg_pixel-R50/checkpoint.pt
 
python tti/read_coco_results_tool.py -i logs/$out_dir/
./exp2.sh















  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_gt_split_but_train_scoring/R50 \
  --exp_str regular_5-sgdt_1_gt_split_but_train_scoring \
  --coco_path coco \
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
  --resume $out_dir/checkpoint.pth
  
  
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50 \
  --exp_str gt_only_exp-no_split-pad32-debug_gt_remove_ratio0.5 \
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
  
  
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50 \
  --exp_str gt_all_tokens_with_random_order_pad16 \
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
  --resume logs/$out_dir/R50/checkpoint.pth

#===========================
pad_fg_pixel=16
token_scoring_loss_criterion=reg_sigmoid_l1_loss
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=pred_significance-reclaim_padded_region-no_bg_token_remove-split_sig_thd0.5-filter_false_split
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
  --output_dir logs/regular_5-sgdt_1_dynamic_vit_pred_fg_only_remove_0.1_wo_gf_pad_fg_pixel_16/R50 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  dynamic_vit_pred_fg_only_remove_0.1_wo_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_dynamic_vit_pred_fg_only_remove_0.3_wo_gf_pad_fg_pixel_16/R50 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  dynamic_vit_pred_fg_only_remove_0.3_wo_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 


python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_pred_fg_only_remove_0.1_wo_gf_0.1/R50 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  pred_fg_only_remove_0.1_wo_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 
  
  
  
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_gt_split_but_train_scoring/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  test_but_scoring_grad_for_loss_only  \
  --token_scoring_loss_criterion reg_sigmoid_l1_loss  \
  --token_scoring_gt_criterion significance_value \
  --wandb \
  --resume logs/regular_5-sgdt_1_gt_split_but_train_scoring/R50/checkpoint.pth
  
  
  
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_regular_5-sgdt_1_scoring_differentiable_with_supervision_ce_independent/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  v1_selection_differentiable  \
  --token_scoring_loss_criterion fg_scale_class_pred_ce_independent  \
  --token_scoring_gt_criterion significance_value \
  --wandb 
  
  
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_5sgdt1_small_scale_with_priority_gt_token_label/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 5  \
  --num_encoder_sgdt_layers 1 \
  --fg_scale_supervision \
  --token_scoring_version  v0_with_gt \
  --wandb \
  --resume logs/sgdt_dn_dab_detr_lr0.5_x2gpus_5sgdt1_small_scale_with_priority_gt_token_label/R50/checkpoint.pth

2022-8-1

  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_adapt_with_supervision_dn_dab_detr_lr0.25_x2gpus/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 2.5e-5 \
  --lr_backbone 2.5e-6 \
  --find_unused_params \
  --num_encoder_sgdt_layers 1  \
  --enc_layers 5 \
  --wandb \
  --resume logs/sgdt_adapt_with_supervision_dn_dab_detr_lr0.25_x2gpus/R50/checkpoint.pth





  python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port\
  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_adapt_no_supervision_dn_dab_detr_lr0.25_x2gpus/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 2.5e-5 \
  --lr_backbone 2.5e-6 \
  --find_unused_params \
  --num_encoder_sgdt_layers 1  \
  --enc_layers 5
    
  --resume logs/sgdt_only_last_layer_dn_dab_detr_lr0.25_x2gpus/R50/checkpoint.pth
 python main.py -m sgdt_dn_dab_detr \
  --output_dir logs/debug_sgdt_dn_dab_detr_lr0.25/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 2.5e-5 \
  --lr_backbone 2.5e-6 \
  --resume logs/debug_sgdt_dn_dab_detr_lr0.25/R50/checkpoint.pth



\
  --resume logs/debug_sgdt_dn_dab_detr_lr0.25/R50/checkpoint.pth
  
  2022-8-7
  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_token_wo_su_after_debug_lr0.5_x2gpus/R50 \
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
  --wandb \
  --resume logs/sgdt_token_wo_su_after_debug_lr0.5_x2gpus/R50/checkpoint.pth
  2022-8-9
  
  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/dn_dab_detr_lr0.5_x2gpus_encoder0layer_w_pos/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 0  \
  --num_encoder_sgdt_layers 0 \
  --wandb 
  
  
  2022-8-10
    main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_ec_5l_sgdtv1_1l_with_gt_token_label/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --enc_layers 5  \
  --num_encoder_sgdt_layers 1 \
  --encoder_sgdt_layer_version 1 \
  --fg_scale_supervision \
  --wandb 
  
COMMENT
