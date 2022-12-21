#!/usr/bin/env bash
#source ./utils.sh
export TORCH_DISTRIBUTED_DEBUG=INFO # or DETAIL
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
log_date=$(date +"%y-%m-%d-%H-%M-%S") # -%M-%S
GPU_NUM=2
#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************
master_port=29500


eval_model_with_proposal() {
	pad_fg_pixel_=$1
	token_scoring_loss_criterion_=$2
	token_scoring_gt_criterion_=$3
	token_scoring_discard_split_criterion_=$4
	weight_dir_=$5
	batch_size_=$6
	exp_str_=$7

	result_dir_=results_/test_results-$exp_str_
	mkdir $result_dir_
	log_date_=$(date +"%y-%m-%d-%H-%M-%S") # -%M-%S
	python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
	main.py -m sgdt_dn_dab_detr \
	  --output_dir $weight_dir_ \
	  --batch_size $batch_size_ \
	  --epochs 12 \
	  --lr_drop 11 \
	  --coco_path coco \
	  --use_dn \
	  --lr 5e-5 \
	  --lr_backbone 5e-6 \
	  --encoder_layer_config regular_5-sgdt_1 \
	  --token_scoring_discard_split_criterion  $token_scoring_discard_split_criterion_  \
	  --token_scoring_loss_criterion $token_scoring_loss_criterion_  \
	  --token_scoring_gt_criterion $token_scoring_gt_criterion_ \
	  --pad_fg_pixel $pad_fg_pixel_ \
	  --sgdt_transformer \
	  --proposal_scoring min_split_score0.3-nms_thd0.6  \
	  --use_decoder_proposal \
	  --eval \
	  --pretrain_model_path  $weight_dir_/checkpoint.pth | tee $result_dir_/"$log_date_"batchsize-$batch_size_-$token_scoring_discard_split_criterion_-$token_scoring_gt_criterion_
}
#-m sgdt_dn_dab_detr   --output_dir logs/ttt  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1    --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value --token_scoring_discard_split_criterion gt_only_exp-no_bg_token_remove-reclaim_padded_region  --eval --resume logs/regular_5-sgdt_1/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-V-K-same-src/checkpoint.pth --debug
#===========================


pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1
exp_str=lr0.1_finetune-sgdt_transformer-use_decoder_proposal
weight_dir=logs/$out_dir/$exp_str

batch_size=2
eval_model_with_proposal $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size $exp_str

batch_size=4
eval_model_with_proposal $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size $exp_str


batch_size=8
eval_model_with_proposal $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size $exp_str
  
  
#./dist_train1.sh
  

<<COMMENT

pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-use_proposal
out_dir=regular_5-sgdt_1
weight_dir=logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad0-bilinear
exp_str=finetune-split-use_proposal-min_split_score0.3-nms_thd0.6




pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
# token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_gt_split_ratio0.0
# -num_tokens_to_increase50
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1


weight_dir=logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-V-K-same-src

batch_size=4
eval_model $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size


batch_size=8
  eval_model $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size
  
  

pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
# token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_gt_split_ratio0.0
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=regular_5-sgdt_1


weight_dir=logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-V-K-same-src
batch_size=2
weight_dir=logs/$out_dir/R50
batch_size=2

pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value_inverse_fg
# token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_gt_split_ratio0.0
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region
out_dir=benchmark-version/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value_inverse_fg


#===========================
  eval_model $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size


batch_size=4
  eval_model $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size



batch_size=1
  eval_model $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size
  
batch_size=8
  eval_model $pad_fg_pixel $token_scoring_loss_criterion $token_scoring_gt_criterion $token_scoring_discard_split_criterion $weight_dir 	$batch_size



python -m torch.distributed.launch --nproc_per_node=1 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-V-K-same-src \
  --exp_str   gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16 \
  --batch_size 4 \
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
  --resume logs/$out_dir/R50-regular_5-sgdt_1-gt_only_exp-no_bg_token_remove-reclaim_padded_region-pad16-V-K-same-src/checkpoint.pth
  



















#===========================
pad_fg_pixel=32
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg  #
token_scoring_discard_split_criterion=gt_only_exp
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

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
  --resume logs/gt_only_exp-32-gt_fg_scale_fake-fg_scale_class_all_fg/R50/back/checkpoint.pth





pad_fg_pixel=32
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=gt_only_exp-no_split-use_proposal-proposal_remove_gt_filter
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

  python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/with-proposal-R50-regular_5-sgdtv1_1-no_split \
  --exp_str min_fg_score0.05-min_split_score0.5-nms_thd0.6-no_split \
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
  
python tti/read_coco_results_tool.py -i logs/$out_dir/



./exp1.sh

# to do
# share the split in multi sgdt layers 


pad_fg_pixel=32
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=gt_only_exp-no_split
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

  python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/regular_5-sgdtv1_1-R50 \
  --exp_str regular_5-sgdtv1_1-gt_only_exp-no_split-pad32-debug_gt_remove_ratio0.5 \
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
  --resume logs/$out_dir/regular_5-sgdtv1_1-R50/checkpoint.pth
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50 \
  --exp_str gt_only_exp-no_split-pad32 \
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
  --wandb  #\
  #--resume logs/$out_dir/R50/checkpoint.pth
  
  
pad_fg_pixel=16
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=fg_scale_class_all_fg
token_scoring_discard_split_criterion=gt_only_exp-reclaim_padded_region-no_bg_token_remove
out_dir=$token_scoring_discard_split_criterion"-"$pad_fg_pixel"-"$token_scoring_loss_criterion"-"$token_scoring_gt_criterion
#===========================

python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/$out_dir/R50 \
  --exp_str gt_all_fg_with_random_order_pad16 \
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
  
  
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/conv_pred_fg_only_remove-bg_sig_thd0.7-debug/R50 \
  --batch_size 2 \
  --epochs 12 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  conv_pred_fg_only_remove-bg_sig_thd0.7-debug  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb \
  --resume logs/conv_pred_fg_only_remove-bg_sig_thd0.7-debug/R50/checkpoint.pth
  
  

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_pred_fg_only_remove_0.5_wo_gf_debug/R50 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  pred_fg_only_remove_0.5_wo_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 
  
  
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_pred_fg_only_remove_0.3_wo_gf_debug/R50 \
  --batch_size 2 \
  --epochs 2 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  pred_fg_only_remove_0.3_wo_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 
python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \
main.py -m sgdt_dn_dab_detr \
  --output_dir logs/regular_5-sgdt_1_dynamic_vit_pred_fg_only_remove_0.1_w_gf_pad_fg_pixel_16/R50 \
  --batch_size 2 \
  --epochs 4 \
  --lr_drop 11 \
  --coco_path coco \
  --use_dn \
  --lr 5e-5 \
  --lr_backbone 5e-6 \
  --encoder_layer_config regular_5-sgdt_1 \
  --token_scoring_discard_split_criterion  dynamic_vit_pred_fg_only_remove_0.1_w_gf  \
  --token_scoring_loss_criterion fg_weighted_ce  \
  --token_scoring_gt_criterion significance_value \
  --pad_fg_pixel 16 \
  --wandb 


main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x2gpus_regular_5-sgdt_1_v1_selection_differentiable/R50 \
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
  --wandb \
  --resume logs/sgdt_dn_dab_detr_lr0.5_x2gpus_regular_5-sgdt_1_v1_selection_differentiable/R50/checkpoint.pth

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
  --fg_scale_supervision \
  --wandb \
  --resume logs/sgdt_dn_dab_detr_lr0.5_x2gpus_ec_sgdt_only_v1_multi_6l_with_gt_token_label/R50/checkpoint.pth
  
  
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
  --fg_scale_supervision \
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
  --fg_scale_supervision \
  --wandb
COMMENT


