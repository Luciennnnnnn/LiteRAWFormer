# !/bin/bash


# The following code will use all available visible GPUs.
# accelerate launch --multi_gpu \
# --main_process_port 41000 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# inference.py --input_dir ./datasets/RAWSR/test_in --output_dir ./results/RAWSR/test_out --model_dir ./pretrained_models/NTIRE2025_finnal \
# --self_ensemble self horizontal_flip vertical_flip rot90 rot180 rot270

# accelerate launch --num_processes=1 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_efficiency.py --input_dir ./datasets/RAWSR/val_in --output_dir ./results/RAWSR/val_out --model_dir ./pretrained_models/NTIRE2025_finnal \
# --self_ensemble self

accelerate launch --num_processes=1 \
--num_machines 1 \
--mixed_precision no \
--dynamo_backend no \
measure_efficiency.py --input_dir ./datasets/RAWSR/val_in --output_dir ./results/RAWSR/val_out_fp16 --model_dir ./pretrained_models/NTIRE2025_finnal \
--self_ensemble self --dtype fp16

# accelerate launch --num_processes=1 \
# --num_machines 1 \
# --mixed_precision no \
# --dynamo_backend no \
# measure_efficiency.py --input_dir ./datasets/RAWSR/val_in --output_dir ./results/RAWSR/val_out --model_dir ./pretrained_models/NTIRE2025_finnal \
# --self_ensemble self horizontal_flip vertical_flip rot90 rot180 rot270

# RAW outputs: ./results/RAWSR/test_out/raw
# RGB outputs: ./results/RAWSR/test_out/rgb