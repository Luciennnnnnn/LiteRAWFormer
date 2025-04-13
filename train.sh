# !/bin/bash

experiments=(
rawsr_LiteRAWFormer_ps256_bs8
)

for((i=0;i<${#experiments[@]};i++))
do
    accelerate launch --multi_gpu \
    --main_process_port 41001 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    train.py --config options/train/${experiments[i]}.yml
done