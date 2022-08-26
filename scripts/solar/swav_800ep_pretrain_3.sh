#!/bin/bash

DATASET_PATH="/scratch/zdc6/data/solar-large/"
EXPERIMENT_PATH="./experiments/solar/swav_800ep_pretrain_3"
mkdir -p $EXPERIMENT_PATH

NCCL_DEBUG="INFO"
 
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
    --data_path $DATASET_PATH \
    --nmb_crops 2 6 \
    --size_crops 224 96 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.14 \
    --queue_length 3584 \
    --epochs 800 \
    --batch_size 64 \
    --base_lr 0.6 \
    --final_lr 0.0006 \
    --freeze_prototypes_niters 2504 \
    --warmup_epochs 10 \
    --start_warmup 0.3 \
    --task solar \
    --use_fp16 true \
    --dump_path $EXPERIMENT_PATH