# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

train_data_dirs=(
#     $WORK/data/improved/base_mixedp_ic16/train
#     $WORK/data/improved/skew_mixedp_ic16/train
    $WORK/data/improved/final_skew40/train
    $WORK/data/improved/final_skew40/train_z5_z10
    $WORK/data/improved/final_base40/train
    $WORK/data/improved/final_base40/train_z5_z10
)
train_data_dirs_json=$(printf '%s\n' "${train_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "train_data_dirs: $train_data_dirs_json"

ulimit -n 99999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 6 ))

        # CUDA_DEVICES=0,1,2,3
        CUDA_DEVICES=2,3,4,5,6,7
        NUM_DEVICES=$(echo "$CUDA_DEVICES" | tr -d ' ' | tr ',' '\n' | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29504 \
                scripts/chronos/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                chronos.model_id="amazon/chronos-t5-small" \
                chronos.model_type=seq2seq \
                chronos.random_init=false \
                chronos.tie_embeddings=true \
                chronos.context_length=512 \
                chronos.prediction_length=128 \
                chronos.num_samples=20 \
                chronos.n_tokens=4096 \
                chronos.n_special_tokens=2 \
                chronos.pad_token_id=0 \
                chronos.eos_token_id=1 \
                chronos.use_eos_token=true \
                chronos.tokenizer_class=MeanScaleUniformBins \
                chronos.tokenizer_kwargs.low_limit=-15.0 \
                chronos.tokenizer_kwargs.high_limit=15.0 \
                chronos.temperature=1.0 \
                chronos.top_k=50 \
                chronos.top_p=1.0 \
                train.max_steps=400_000 \
                train.save_steps=50_000 \
                train.log_steps=1000 \
                shuffle_buffer_length=100_000 \
                train.per_device_train_batch_size=100 \
                train.warmup_ratio=0.05 \
                train.torch_compile=true \
                train.weight_decay=0.0 \
                train.output_dir=$WORK/checkpoints/ \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/chronos/train.py \
                run_name=DEBUG \
                shuffle_buffer_length=100 \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi
