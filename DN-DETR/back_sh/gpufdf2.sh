#!/usr/bin/env bash


export CUDA_VISIBLE_DEVICES=2
#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=1

python  main.py -m sgdt_dn_dab_detr \
  --output_dir logs/sgdt_dn_dab_detr_lr0.5_x1gpus_b4_e5l_sgdt1l_pred_score_sumbel_softmax/R50 \
  --batch_size 4 \
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


./gpu2_add.sh

<<COMMENT
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
  
COMMENT


