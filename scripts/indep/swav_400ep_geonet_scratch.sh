#!/bin/bash

DATASET_PATH="/scratch/sl636/checked_image/"
EXPERIMENT_PATH="./experiments/indep/swav/geonet_from_scratch_400/"

mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path $DATASET_PATH \
--task geonet_1M_checked \
--initialize_imagenet false \
--project geonet_from_scratch \
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
--batch_size 32 \
--base_lr 0.6 \
--final_lr 0.0006 \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--sync_bn pytorch \
--use_fp16 true \
--dump_path $EXPERIMENT_PATH
