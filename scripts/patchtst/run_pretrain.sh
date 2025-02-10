# On multiple GPUs (example with 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
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
        patchtst.mode=pretrain \
        patchtst.max_wavelength=500 \
        patchtst.rope_percent=0.75 \
        patchtst.clamp_low=-15.0 \
        patchtst.clamp_high=15.0 \
        patchtst.loss=huber \
        patchtst.huber_delta=10.0 \
        train.per_device_train_batch_size=512 \
        train.max_steps=100_000 \
        train.save_steps=50_000 \
        train.log_steps=1_000 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=1e-4 \
        noiser.enabled=false \
        noiser.schedule_name=cosine \
        noiser.start=1.0 \
        noiser.end=0.0 \
        noiser.eps=0.008 \
        noiser.epoch_stop=0.5 \
        noiser.log_steps=100 \
        fixed_dim=3 \
        use_quadratic_embedding=false \
        "$@"
