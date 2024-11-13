#!/bin/bash

DATASET_PATH="/scratch/sl636/ILSVRC/Data/CLS-LOC/"
DUMP_PATH="./experiments/indep/swav/eval_linear_geonet_400/"
MODEL_PATH="/shared/data/swav_models/geonet_from_scratch_400/checkpoints/compatible_ckp-399.pth"

mkdir -p $DUMP_PATH

python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py \
--dump_path $DUMP_PATH \
--data_path $DATASET_PATH \
--pretrained $MODEL_PATH \

