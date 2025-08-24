#!/bin/bash
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
    $WORK/data/final_skew40/train_z5_z10
    $WORK/data/final_base40/train
    $WORK/data/final_base40/train_z5_z10
)
train_data_dirs_json=$(printf '%s\n' "${train_data_dirs[@]}" | jq -R . | jq -s -c .)

ulimit -n 999999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 4 ))

        # CUDA_DEVICES=0,1,2,3
        CUDA_DEVICES=4,5,6,7
        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)
        echo "CUDA_DEVICES: $CUDA_DEVICES"
        echo "NUM_DEVICES: $NUM_DEVICES"
        echo "CORES_PER_JOB: $CORES_PER_JOB"
        echo "CORES_PER_GROUP: $CORES_PER_GROUP"
        echo "TOTAL_CORES: $TOTAL_CORES"

        if python -c "import torch; print(torch.version.hip)" 2>/dev/null | grep -vq "None"; then
            # ROCm detected - disable P2P
            export NCCL_P2P_DISABLE=1
        fi
        export PYTHONWARNINGS="ignore::FutureWarning"

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node $NUM_DEVICES \
                --standalone \
                scripts/patchtst/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                patchtst.mode=predict \
                patchtst.use_dynamics_embedding=false \
                patchtst.poly_degrees=2 \
                patchtst.num_poly_feats=120 \
                patchtst.poly_degrees=2 \
                patchtst.rff_trainable=false \
                patchtst.rff_scale=1.0 \
                patchtst.num_rff=256 \
                patchtst.pretrained_encoder_path=null \
                patchtst.context_length=512 \
                patchtst.prediction_length=128 \
                patchtst.patch_length=16 \
                patchtst.patch_stride=16 \
                patchtst.num_hidden_layers=8 \
                patchtst.num_attention_heads=8 \
                patchtst.d_model=512 \
                patchtst.ffn_dim=512 \
                patchtst.num_rff=256 \
                patchtst.rff_scale=1.0 \
                patchtst.rff_trainable=false \
                patchtst.num_poly_feats=120 \
                patchtst.poly_degrees=2 \
                patchtst.norm_type=rmsnorm \
                patchtst.channel_attention=true \
                patchtst.max_wavelength=500 \
                patchtst.rope_percent=0.75 \
                patchtst.pooling_type=mean \
                patchtst.loss=mse \
                patchtst.distribution_output=null \
                train.per_device_train_batch_size=1024 \
                train.max_steps=100_000 \
                train.save_steps=20_000 \
                train.log_steps=1_000 \
                train.warmup_ratio=0.1 \
                train.torch_compile=true \
                train.weight_decay=0.0 \
                train.output_dir=$WORK/checkpoints/ \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/patchtst/train.py \
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
