#!/bin/bash

DATASET_PATH="/develop/data/0k_251k_uint8_jpeg_tif/"
EXPERIMENT_PATH="/develop/results/ssl4eo_from_scratch_400/"

mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--data_path $DATASET_PATH \
--task ssl4eo \
--initialize_imagenet false \
--project ssl4e0 \
--workers 32 \
--restrict_gpus 0 1 2 3 \
--wandb false \
--nmb_crops 2 6 \
--size_crops 224 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 3840 \
--epoch_queue_starts 15 \
--epochs 400 \
--checkpoint_freq 50 \
--batch_size 64 \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH
