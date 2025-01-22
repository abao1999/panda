# On multiple GPUs (example with 4 GPUs)
# /stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-316/checkpoint-final
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
        shuffle_buffer_length=100_000 \
        patchtst.pretrained_encoder_path=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-367/checkpoint-final \
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
        patchtst.loss=huber \
        patchtst.huber_delta=10.0 \
        patchtst.clamp_low=-15.0 \
        patchtst.clamp_high=15.0 \
        patchtst.distribution_output=null \
        train.per_device_train_batch_size=512 \
        train.max_steps=200_000 \
        train.save_steps=100_000 \
        train.log_steps=1_000 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=1e-4 \
        noiser.enabled=false \
        fixed_dim=3 \
        use_quadratic_embedding=false \
        "$@"

# TODO: try playing with: patch_stride, pre_norm, pooling_type, dropout, head_dropout, attention_dropout, positional_dropout, ff_dropout, norm_type, mask_type