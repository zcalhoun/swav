#!/bin/bash

DATASET_PATH="/scratch/zdc6/data/crop_delineation"
EXPERIMENT_PATH="./experiments/cropdelineation/swav_200ep_pretrain"
mkdir -p $EXPERIMENT_PATH

NCCL_DEBUG="INFO"
 
python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
    --data_path $DATASET_PATH \
    --nmb_crops 2 6 \
    --size_crops 160 96 \
    --min_scale_crops 0.14 0.05 \
    --max_scale_crops 1. 0.14 \
    --queue_length 0 \
    --epochs 400 \
    --batch_size 64 \
    --nmb_prototypes 100 \
    --base_lr 0.4 \
    --final_lr 0.000004 \
    --freeze_prototypes_niters 5005 \
    --warmup_epochs 0 \
    --task crop-delineation \
    --dump_path $EXPERIMENT_PATH