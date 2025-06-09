# !/bin/bash

experiments=(
rawsr_LiteRAWFormer_ps256_bs8
)

RANK=${1:-0}  # 如果 $1 为空，则使用 "default1"
MASTER_ADDR=${2:-1}  # 如果 $2 为空，则使用 "default2"
MASTER_PORT=${3:-29500}
WORLD_SIZE=${4:-1}

for((i=0;i<${#experiments[@]};i++))
do
    accelerate launch  \
    --machine_rank=$RANK \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --num_machines=$WORLD_SIZE \
    --use_fsdp \
    --fsdp_offload_params false \
    --fsdp_sharding_strategy HYBRID_SHARD_ZERO2 \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap TransformerBlock \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --fsdp_forward_prefetch false \
    --fsdp_use_orig_params True \
    --fsdp_cpu_ram_efficient_loading false \
    --fsdp_sync_module_states True \
    train_fsdp.py --config options/train/${experiments[i]}.yml
    # --fsdp_state_dict_type SHARDED_STATE_DICT \
done