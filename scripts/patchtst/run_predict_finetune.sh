# read debug flag
DEBUG=0
while getopts "d" flag; do
        case "${flag}" in
                d) DEBUG=1;;
        esac
done
shift $((OPTIND - 1))

ulimit -n 99999
if [ "$DEBUG" -eq 0 ]; then

        TOTAL_CORES=$(nproc)
        CORES_PER_GROUP=$(( $TOTAL_CORES / 2 ))
        CORES_PER_JOB=$(( $CORES_PER_GROUP / 4 ))

        # CUDA_DEVICES=0,1,2,3
        CUDA_DEVICES=4,5,6,7

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$CORES_PER_JOB torchrun \
                --nproc-per-node 4 \
                --master-port 29501 \
                scripts/patchtst/train.py \
                shuffle_buffer_length=100_000 \
                patchtst.mode=predict \
                patchtst.use_dynamics_embedding=false \
                patchtst.pretrained_encoder_path=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/mlm_cc_stand-0/checkpoint-final \
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
                train.save_steps=50_000 \
                train.log_steps=1_000 \
                train.warmup_ratio=0.1 \
                train.torch_compile=true \
                train.weight_decay=0.0 \
                scheduler.enabled=false \
                "$@"
else  # this mode allows for breakpoints inside model code
        CUDA_VISIBLE_DEVICES=0 python scripts/patchtst/train.py \
                run_name=DEBUG \
                patchtst.pretrained_encoder_path=$WORK/checkpoints/mlm40_stand-0/checkpoint-final \
                shuffle_buffer_length=100 \
                patchtst.mode=predict \
                train.ddp_backend=null \
                train.torch_compile=false \
                "$@"
fi

