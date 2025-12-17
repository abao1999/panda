#!/bin/bash

# This script trains panda in forecast mode with different patch sizes (and matching patch strides), while maintaining other model parameters
# i.e. keep d_model and ffn_dim constant
# We also disable the dynamics embedding to ensure that we isolate the effect of patch size, as the dynamics embedding will be different for different patch sizes
# NOTE: patch size affects the maximum possible batch size we can use. For example, if we halve the patch size, we double the number of patches, so we can only use half the batch size
# read debug flag

DEBUG=0
while getopts "d" flag; do
    case "${flag}" in
            d) DEBUG=1;;
    esac
done
shift $((OPTIND - 1))

train_data_dirs=(
    $WORK/data/final_skew40/train
    $WORK/data/final_base40/train
)
train_data_dirs_json=$(printf '%s\n' "${train_data_dirs[@]}" | jq -R . | jq -s -c .)

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


    # NOTE: since we run these experiments on the AMD GPUs, our specific setup requires:
    # train.dataloader_num_workers=16 and train.dataloader_prefetch_factor=2

    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun \
            --nproc-per-node $NUM_DEVICES \
            --standalone \
            scripts/panda/train.py \
            shuffle_buffer_length=100_000 \
            train_data_dirs=$train_data_dirs_json \
            patchtst.mode=predict \
            patchtst.use_dynamics_embedding=false \
            patchtst.poly_degrees=2 \
            patchtst.num_poly_feats=120 \
            patchtst.rff_trainable=false \
            patchtst.rff_scale=1.0 \
            patchtst.num_rff=256 \
            patchtst.pretrained_encoder_path=null \
            patchtst.context_length=512 \
            patchtst.prediction_length=128 \
            patchtst.patch_length=32 \
            patchtst.patch_stride=32 \
            patchtst.num_hidden_layers=8 \
            patchtst.num_attention_heads=8 \
            patchtst.d_model=512 \
            patchtst.ffn_dim=512 \
            patchtst.norm_type=rmsnorm \
            patchtst.channel_attention=true \
            patchtst.max_wavelength=500 \
            patchtst.rope_percent=0.75 \
            patchtst.pooling_type=mean \
            patchtst.loss=mse \
            patchtst.distribution_output=null \
            train.per_device_train_batch_size=2048 \
            train.max_steps=100_000 \
            train.save_steps=20_000 \
            train.log_steps=1_000 \
            train.warmup_ratio=0.1 \
            train.torch_compile=true \
            train.weight_decay=0.0 \
            train.output_dir=$WORK/checkpoints/ \
            train.dataloader_num_workers=16 \
            train.dataloader_prefetch_factor=2 \
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
