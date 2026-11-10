#!/bin/bash
# exp5 训练脚本: DINOv2-B/14 + 因果解耦融合 + UPerNet
# 与 exp2 对齐: GPU1, batch16, backbone_lr=1e-5, head_lr=1e-4

export CUDA_VISIBLE_DEVICES=0


python3 train.py \
    --data_root ./data/train_ready \
    --base_size 518 \
    --crop_size 518 \
    --num_classes 10 \
    --arch vit_base \
    --clip_model_name ViT-B/16 \
    --beta_x 0.5 \
    --beta_p 0.5 \
    --beta_b 0.1 \
    --batch_size 16 \
    --epochs 50 \
    --backbone_lr 1e-5 \
    --head_lr 1e-4 \
    --fusion_lr 5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 3 \
    --patience 8 \
    --use_text_fusion \
    --text_prompts_file ./mars_prompts.txt \
    --checkpoint_dir ./checkpoints \
    --log_dir ./logs \
    --device cuda \
    --seed 42
