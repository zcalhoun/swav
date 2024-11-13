#!/bin/bash

DATASET_PATH="/scratch/sl636/inat/"
DUMP_PATH="./experiments/indep/swav/eval_linear_inat/"
MODEL_PATH="/home/sl636/swav/experiments/indep/swav/imagenet_from_scratch_400/checkpoints/ckp-399.pth"

mkdir -p $DUMP_PATH

python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--dump_path $DUMP_PATH \
--data_path $DATASET_PATH \
--pretrained $MODEL_PATH \
--epochs 84 \
--lr 0.01 \
--wd 0.0001 \
--batch_size 256 \
--scheduler_type step \
--decay_epochs 28 56 84 \
--gamma 0.1 \
