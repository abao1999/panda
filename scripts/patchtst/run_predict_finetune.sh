# On multiple GPUs (example with 4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
        shuffle_buffer_length=100_000 \
        patchtst.pretrained_encoder_path=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-281/checkpoint-final \
        patchtst.context_length=512 \
        patchtst.prediction_length=128 \
        patchtst.patch_length=16 \
        patchtst.patch_stride=16 \
        patchtst.num_hidden_layers=8 \
        patchtst.num_attention_heads=8 \
        patchtst.d_model=512 \
        patchtst.quantizer_high=15.0 \
        patchtst.quantizer_low=-15.0 \
        patchtst.norm_type=rmsnorm \
        patchtst.channel_attention=true \
        patchtst.use_channel_embedding=true \
        patchtst.channel_embedding=quadratic \
        patchtst.mode=predict \
        train.per_device_train_batch_size=256 \
        train.max_steps=300_000 \
        train.save_steps=100_000 \
        train.log_steps=1_000 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=1e-4 \
        quantizer.enabled=false \
        noiser.enabled=false \
        use_time_delay_embed=false \
        fixed_dim=3 \
        "$@"
