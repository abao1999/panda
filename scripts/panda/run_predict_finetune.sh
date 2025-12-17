#!/bin/bash

# This script runs the panda training (predict mode) on the specified train_data_dirs, providing option for either:
# 1. finetuning an MLM checkpoint, or
# 2. training in predict mode from scratch

# read debug flag
DEBUG=0
while getopts "d" flag; do
    case "${flag}" in
            d) DEBUG=1;;
    esac
done
shift $((OPTIND - 1))

# NOTE: all scaling up runs use the improved data (can be verified by checking wandb project configs)
# train_data_dirs=(
#     $WORK/data/improved/final_skew40/train
#     $WORK/data/improved/final_base40/train
# )

train_data_dirs=(
    $WORK/data/improved/base_mixedp_ic16/train # ics 1-8
    $WORK/data/improved/skew_mixedp_ic16/train # ics 1-8
    $WORK/data/improved/base_mixedp_ic16/train_more # ics 9-16
    $WORK/data/improved/skew_mixedp_ic16/train_more # ics 9-16
)
train_data_dirs_json=$(printf '%s\n' "${train_data_dirs[@]}" | jq -R . | jq -s -c .)

echo "train_data_dirs: $train_data_dirs_json"

ulimit -n 999999
if [ "$DEBUG" -eq 0 ]; then

    # CUDA_DEVICES=0,1,2,3
    CUDA_DEVICES=4,5,6,7
    echo "CUDA Devices: $CUDA_DEVICES"

    NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)
    TOTAL_PROCESSES=$(nproc)
    PER_GROUP=$(( $TOTAL_PROCESSES / 2 ))
    PER_RANK=$(( $PER_GROUP / $NUM_DEVICES ))

    export OMP_NUM_THREADS=$PER_RANK
    export MKL_NUM_THREADS=$PER_RANK
    export OPENBLAS_NUM_THREADS=$PER_RANK
    export NUMEXPR_NUM_THREADS=$PER_RANK

    if python -c "import torch; print(torch.version.hip)" 2>/dev/null | grep -vq "None"; then
        echo "ROCm detected - disabling P2P for distributed training"
        export NCCL_P2P_DISABLE=1
    fi
    export PYTHONWARNINGS="ignore"

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun \
            --nproc-per-node $NUM_DEVICES \
            --standalone \
            scripts/panda/train.py \
            shuffle_buffer_length=10_000 \
            train_data_dirs=$train_data_dirs_json \
            patchtst.mode=predict \
            patchtst.use_dynamics_embedding=true \
            patchtst.poly_degrees=2 \
            patchtst.num_poly_feats=156 \
            patchtst.rff_trainable=false \
            patchtst.rff_scale=1.0 \
            patchtst.num_rff=312 \
            patchtst.pretrained_encoder_path=null \
            patchtst.context_length=512 \
            patchtst.prediction_length=128 \
            patchtst.patch_length=16 \
            patchtst.patch_stride=16 \
            patchtst.num_hidden_layers=10 \
            patchtst.num_attention_heads=10 \
            patchtst.d_model=640 \
            patchtst.ffn_dim=640 \
            patchtst.norm_type=rmsnorm \
            patchtst.channel_attention=true \
            patchtst.max_wavelength=500 \
            patchtst.rope_percent=0.75 \
            patchtst.pooling_type=mean \
            patchtst.loss=mse \
            patchtst.distribution_output=null \
            train.per_device_train_batch_size=512 \
            train.max_steps=1_000_000 \
            train.save_steps=50_000 \
            train.log_steps=1_000 \
            train.warmup_ratio=0.05 \
            train.torch_compile=true \
            train.weight_decay=0.0 \
            train.output_dir=$WORK/checkpoints/ \
            "$@"
else  # this mode allows for breakpoints inside model code
    CUDA_VISIBLE_DEVICES=0 python scripts/panda/train.py \
            run_name=DEBUG \
            train_data_dirs=$train_data_dirs_json \
            patchtst.pretrained_encoder_path=null \
            shuffle_buffer_length=100 \
            patchtst.mode=predict \
            train.ddp_backend=null \
            train.torch_compile=false \
            train.output_dir=$WORK/checkpoints/ \
            "$@"
fi
