# !/bin/bash

experiments=(
rawsr_ps256_bs8
)

for((i=0;i<${#experiments[@]};i++))
do
    CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu \
    --main_process_port 41001 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    train.py --config options/train/${experiments[i]}.yml
done