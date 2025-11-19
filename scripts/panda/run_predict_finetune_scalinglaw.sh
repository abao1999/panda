# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

scalinglaw_data_dir=$WORK/data/improved/scalinglaw

# split_0-163_ic128
# split_163-327_ic64
# split_327-655_ic32
# split_655-1311_ic16
# split_1311-2622_ic8
# split_2622-5244_ic4
# split_5244-10489_ic2

train_data_dirs=(
#     $scalinglaw_data_dir/split_163-327_ic64/train
    $WORK/data/improved/final_skew40/train
)
train_data_dirs_json=$(printf '%s\n' "${train_data_dirs[@]}" | jq -R . | jq -s -c .)
echo "train_data_dirs: $train_data_dirs_json"


checkpoint_dir=$WORK/checkpoints
# chattn_mlm_sys5245_ic4-0
# chattn_mlm_sys164_ic128-1
# chattn_mlm_sys656_ic32-1
# chattn_mlm_sys10490_ic2-0
# chattn_mlm_sys1312_ic16-0
# chattn_mlm_sys2623_ic8-1
# chattn_mlm_sys328_ic64-1
# chattn_mlm_sys20k_ic1-0

checkpoint_name=chattn_mlm_sys20k_ic1-0
checkpoint_path=$checkpoint_dir/$checkpoint_name/checkpoint-final
echo "checkpoint_path: $checkpoint_path"

ulimit -n 99999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 3 ))

        # CUDA_DEVICES=0,1,2
        CUDA_DEVICES=5,6,7
        NUM_DEVICES=$(tr ',' '\n' <<< "$CUDA_DEVICES" | wc -l)

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node $NUM_DEVICES \
                --master-port 29501 \
                scripts/panda/train.py \
                shuffle_buffer_length=100_000 \
                train_data_dirs=$train_data_dirs_json \
                patchtst.mode=predict \
                patchtst.use_dynamics_embedding=false \
                patchtst.pretrained_encoder_path=$checkpoint_path \
                patchtst.context_length=512 \
                patchtst.prediction_length=128 \
                patchtst.patch_length=16 \
                patchtst.patch_stride=16 \
                patchtst.num_hidden_layers=8 \
                patchtst.num_attention_heads=8 \
                patchtst.d_model=512 \
                patchtst.norm_type=rmsnorm \
                patchtst.channel_attention=true \
                patchtst.max_wavelength=500 \
                patchtst.rope_percent=0.75 \
                patchtst.pooling_type=mean \
                patchtst.loss=mse \
                patchtst.distribution_output=null \
                train.per_device_train_batch_size=1024 \
                train.max_steps=100_000 \
                train.save_steps=10_000 \
                train.log_steps=1_000 \
                train.warmup_ratio=0.1 \
                train.torch_compile=true \
                train.weight_decay=0.0 \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/panda/train.py \
                run_name=DEBUG \
                patchtst.pretrained_encoder_path=$WORK/checkpoints/mlm40_stand-0/checkpoint-final \
                shuffle_buffer_length=100 \
                patchtst.mode=predict \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi

