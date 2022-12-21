#!/usr/bin/env bash
#source ./utils.sh

export CUDA_VISIBLE_DEVICES=0,1
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1
GPU_NUM=2
#*****************************************
#Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being #overloaded, please further tune the variable for optimal performance in your application as needed. 
#*****************************************
master_port=29500  
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=$master_port \

python main.py -m sgdt_dn_dab_detr --output_dir logs/dn_dab_detr/tttt  --batch_size 2 --epochs 12 --lr_drop 11 --coco_path coco --use_dn  --lr 2.5e-5 --lr_backbone 2.5e-6  --encoder_layer_config regular_5-sgdt_1 --token_scoring_discard_split_criterion pred_significance-reclaim_padded_region-no_bg_token_remove-split_sig_thd0.5-filter_false_split  --token_scoring_loss_criterion reg_sigmoid_l1_loss  --token_scoring_gt_criterion significance_value --token_adaption_visualization  --resume logs/debug/pred_significance-reclaim_padded_region-no_bg_token_remove-split_sig_thd0.5-filter_false_split-16-reg_sigmoid_l1_loss-significance_value/R50/checkpoint.pth --pad_fg_pixel 16
  


<<COMMENT
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


