#!/bin/bash

DATASET_PATH="/scratch/sl636/EuroSAT_final/"
EXPERIMENT_PATH="./experiments/indep/swav/geonet_eurosat_5/"
PRETRAINED_PATH="/home/sl636/swav/experiments/indep/swav/geonet_from_scratch_400/checkpoints/compatible_ckp-399.pth"

mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
--data_path $DATASET_PATH \
--task eurosat \
--initialize_imagenet false \
--initialize_pretrained true \
--pretrained_path $PRETRAINED_PATH \
--project geonet_eurosat_5 \
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
--queue_length 0 \
--epoch_queue_starts 15 \
--epochs 5 \
--checkpoint_freq 1 \
--batch_size 32 \
--base_lr 0.0006 \
--final_lr 0.0006 \
--freeze_prototypes_niters 5005 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--use_fp16 true \
--sync_bn pytorch \
--dump_path $EXPERIMENT_PATH
