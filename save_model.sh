#!/bin/bash

MODEL_PATH="/home/sl636/swav/experiments/cropdelineation/swav_800ep_target_pretrain_3000pr/checkpoints/ckp-799.pth"
OUTPUT_PATH="/home/sl636/swav/experiments/cropdelineation/swav_800ep_target_pretrain_3000pr/swav-c2-3000pr.pt"

python save_model.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH 
