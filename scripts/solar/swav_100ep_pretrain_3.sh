#!/bin/bash

DATASET_PATH="/scratch/zdc6/data/"
EXPERIMENT_PATH="./experiments/solar/swav_100ep_pretrain_3"
mkdir -p $EXPERIMENT_PATH

NCCL_DEBUG="INFO"
 
python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
    --data_path $DATASET_PATH \
    --nmb_crops 2 6 \
    --size_crops 224 96 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.14 \
    --queue_length 0 \
    --epochs 200 \
    --batch_size 64 \
    --nmb_prototypes 100 \
    --base_lr 0.8 \
    --final_lr 0.00008 \
    --freeze_prototypes_niters 5005 \
    --warmup_epochs 0 \
    --task solar \
    --dump_path $EXPERIMENT_PATH