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
                --master-port 29500 \
                scripts/patchtst/train.py \
                patchtst.mode=pretrain \
                patchtst.context_length=512 \
                patchtst.patch_length=16 \
                patchtst.patch_stride=16 \
                patchtst.num_hidden_layers=8 \
                patchtst.num_attention_heads=8 \
                patchtst.d_model=512 \
                patchtst.norm_type=rmsnorm \
                patchtst.channel_attention=true \
                patchtst.mask_type=random \
                patchtst.random_mask_ratio=0.5 \
                patchtst.channel_consistent_masking=false \
                patchtst.max_wavelength=500 \
                patchtst.rope_percent=0.75 \
                patchtst.loss=mse \
                patchtst.huber_delta=1.0 \
                train.per_device_train_batch_size=1024 \
                train.max_steps=100_000 \
                train.save_steps=50_000 \
                train.log_steps=1_000 \
                train.warmup_ratio=0.1 \
                train.torch_compile=true \
                scheduler.enabled=true \
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