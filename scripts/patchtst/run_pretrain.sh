# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

train_data_dirs=(
    $WORK/data/improved/base_mixedp_ic16/train
    $WORK/data/improved/skew_mixedp_ic16/train
    $WORK/data/improved/final_skew40/train
    $WORK/data/improved/final_skew40/train_z5_z10
    $WORK/data/improved/final_base40/train
    $WORK/data/improved/final_base40/train_z5_z10
)
train_data_dirs_json=$(printf '%s\n' "${train_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "train_data_dirs: $train_data_dirs_json"

ulimit -n 999999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 6 ))

        CUDA_DEVICES=2,3,4,5,6,7
        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29501 \
                scripts/patchtst/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                patchtst.mode=pretrain \
                patchtst.use_dynamics_embedding=false \
                patchtst.context_length=1024 \
                patchtst.patch_length=16 \
                patchtst.patch_stride=16 \
                patchtst.num_hidden_layers=12 \
                patchtst.num_attention_heads=12 \
                patchtst.d_model=768 \
                patchtst.norm_type=rmsnorm \
                patchtst.channel_attention=true \
                patchtst.mask_type=random \
                patchtst.random_mask_ratio=0.5 \
                patchtst.channel_consistent_masking=false \
                patchtst.max_wavelength=500 \
                patchtst.rope_percent=0.75 \
                patchtst.loss=mse \
                train.per_device_train_batch_size=192 \
                train.max_steps=800_000 \
                train.save_steps=50_000 \
                train.log_steps=1_000 \
                train.warmup_ratio=0.05 \
                train.torch_compile=true \
                scheduler.enabled=false \
                train.output_dir=$WORK/checkpoints/ \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/patchtst/train.py \
                run_name=DEBUG \
                shuffle_buffer_length=100 \
                patchtst.mode=pretrain \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi
