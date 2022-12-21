#!/usr/bin/env bash
#source ./utils.sh

export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
GPU_NUM=3
#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************
master_port=29500  



pad_fg_pixel=0
token_scoring_loss_criterion=gt_fg_scale_fake
token_scoring_gt_criterion=significance_value
token_scoring_discard_split_criterion=gt_only_exp-no_bg_token_remove-reclaim_padded_region-debug_split_1c-token_num_no_limit
out_dir=regular_5-sgdtv1_1
exp_str=token_num_no_limit-feature-distillation-aligh-sgdtv1_1-debug_split_1c
#===========================

python -m torch.distributed.launch --nproc_per_node=$GPU_NUM --master_port=$master_port \
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
  --feature_distillation \
  --resume logs/$out_dir/$exp_str/checkpoint.pth








<<COMMENT
  --eval_decoder_layer 4 \
  
  

for thd in {1..20}
do
    echo "Welcome $thd times"
    #thd=$(($i * 1))
    bc -l thd *= 0.05
    
    python main.py -m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1 --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value --resume logs/debug/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/R50/checkpoint.pth  --eval  --pad_fg_pixel 16 --debug  --token_scoring_discard_split_criterion  gt_only_exp-reclaim_padded_region-no_bg_token_remove-debug_gt_split_ratio$thd
    
done



python main.py -m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion  gt_only_exp-reclaim_padded_region-no_bg_token_remove --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value --resume logs/debug/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/R50/checkpoint.pth  --eval  --pad_fg_pixel 16




--proposal_saved_file logs/proposals_original39.3.pth  --use_proposal 


-m sgdt_dn_dab_detr --output_dir logs/dn_dab_detr/tttt  --batch_size 2 --epochs 12 --lr_drop 11 --coco_path coco --use_dn  --lr 2.5e-5 --lr_backbone 2.5e-6  --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion gt_only_exp --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value  --resume logs/sgdt_dn_dab_detr_lr0.5_x2gpus_5sgdt1_small_scale_with_priority_gt_token_label-v1-47.4/R50/checkpoint.pth  --eval --proposal_saved_file logs/proposals_original39.3.pth  --use_proposal   --token_adaption_visualization

-m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion  gt_only_exp-reclaim_padded_region-no_bg_token_remove --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value --resume logs/debug/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/R50/checkpoint.pth  --eval  --pad_fg_pixel 16 



 --proposal_saved_file logs/proposals_original39.3.pth  --use_proposal    --token_adaption_visualization




-m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion  gt_only_exp-reclaim_padded_region-no_bg_token_remove --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value --resume logs/debug/gt_only_exp-reclaim_padded_region-no_bg_token_remove-16-gt_fg_scale_fake-significance_value/R50/checkpoint.pth  --eval  --pad_fg_pixel 16  --proposal_saved_file logs/proposals_original39.3.pth  --use_proposal  




python main.py -m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion  gt_only_exp --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion significance_value --resume logs/debug/sgdt_dn_dab_detr_lr0.5_x2gpus_5sgdt1_small_scale_with_priority_gt_token_label-v1-47.4/R50/checkpoint.pth  --eval  --pad_fg_pixel 32


--use_proposal --proposal_saved_file logs/proposals_original39.3.pth
 

python main.py -m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion gt_only_exp-reclaim_padded_region-no_bg_token_remove  --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion fake_all_tokens_are_fg --resume logs/R50_lr0.5_x2gpus/checkpoint.pth  --eval --save_results --proposal_saved_file logs/proposals_original39.3.pth


python main.py -m sgdt_dn_dab_detr --output_dir logs/train_eval_results  --batch_size 2 --coco_path coco --use_dn --encoder_layer_config regular_6 --token_scoring_discard_split_criterion gt_only_exp-reclaim_padded_region-no_bg_token_remove  --token_scoring_loss_criterion gt_fg_scale_fake  --token_scoring_gt_criterion fake_all_tokens_are_fg --resume logs/R50_lr0.5_x2gpus/checkpoint.pth  --eval --save_results #--eval_training_data 



 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.658
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.624
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.830



IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.367
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.367
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.816

Process finished with exit code 0


COMMENT


