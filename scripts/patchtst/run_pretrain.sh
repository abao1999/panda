# CUDA_VISIBLE_DEVICES=0 \
# python scripts/patchtst/train.py \
#         run_name=patchtst_lorenz_overfit \
#         patchtst.context_length=512 \
#         patchtst.prediction_length=64 \
#         shuffle_buffer_length=1000 \
#         wandb.log=False \
#         wandb.group_name=finetune_large \
#         train.max_steps=20_000 \
#         train.save_steps=10_000 \
#         train.log_steps=100 \


# On multiple GPUs (example with 4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
        shuffle_buffer_length=100_000 \
        patchtst.context_length=512 \
        patchtst.patch_length=16 \
        patchtst.patch_stride=16 \
        patchtst.num_hidden_layers=8 \
        patchtst.num_attention_heads=8 \
        patchtst.d_model=512 \
        patchtst.norm_type=rmsnorm \
        patchtst.channel_attention=true \
        patchtst.pooling_type=mean \
        patchtst.mask_type=random \
        patchtst.random_mask_ratio=0.5 \
        patchtst.channel_consistent_masking=false \
        patchtst.mode=pretrain \
        patchtst.max_wavelength=500 \
        patchtst.rope_percent=0.75 \
        train.per_device_train_batch_size=64 \
        train.max_steps=100_000 \
        train.save_steps=100_000 \
        train.log_steps=1_000 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=1e-4 \
        noiser.enabled=true \
        noiser.schedule_name=cosine \
        noiser.start=1.0 \
        noiser.end=0.0 \
        noiser.eps=0.008 \
        noiser.epoch_stop=0.5 \
        noiser.log_steps=100 \
        fixed_dim=3 \
        use_quadratic_embedding=false \
        "$@"
