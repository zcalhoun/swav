#!/bin/bash

DATASET_PATH="/scratch/sl636/"
EXPERIMENT_PATH="./experiments/indep/swav/climate+/"

mkdir -p $EXPERIMENT_PATH

python -m torch.distributed.launch --nproc_per_node=4 main_swav.py \
--data_path $DATASET_PATH \
--task climate+ \
--nmb_crops 2 6 \
--size_crops 160 96 \
--min_scale_crops 0.14 0.05 \
--max_scale_crops 1. 0.14 \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--nmb_prototypes 3000 \
--queue_length 0 \
--epochs 200 \
--checkpoint_freq 10 \
--batch_size 256 \
--base_lr 2.4 \
--final_lr 0.0024 \
--wd 0.000001 \
--warmup_epochs 0 \
--arch resnet50 \
--workers 8 \
--use_fp16 true \
--dump_path $EXPERIMENT_PATH
