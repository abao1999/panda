# default mlm: 386
# default ablations mlm: 392
# On multiple GPUs (example with 4 GPUs)
ulimit -n 99999
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
        shuffle_buffer_length=100_000 \
        patchtst.pretrained_encoder_path=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-386/checkpoint-final \
        patchtst.context_length=512 \
        patchtst.prediction_length=128 \
        patchtst.patch_length=16 \
        patchtst.patch_stride=16 \
        patchtst.num_hidden_layers=8 \
        patchtst.num_attention_heads=8 \
        patchtst.d_model=512 \
        patchtst.norm_type=rmsnorm \
        patchtst.channel_attention=true \
        patchtst.mode=predict \
        patchtst.max_wavelength=500 \
        patchtst.rope_percent=0.75 \
        patchtst.pooling_type=mean \
        patchtst.loss=mse \
        patchtst.distribution_output=null \
        train.per_device_train_batch_size=1024 \
        train.max_steps=200_000 \
        train.save_steps=50_000 \
        train.log_steps=1_000 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=0.0 \
        noiser.enabled=true \
        "$@"

