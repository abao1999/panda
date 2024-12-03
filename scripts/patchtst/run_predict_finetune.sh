# On multiple GPUs (example with 4 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
        shuffle_buffer_length=100_000 \
        patchtst.pretrained_encoder_path=/stor/work/AMDG_Gilpin_Summer2024/checkpoints/run-303/checkpoint-final \
        patchtst.context_length=512 \
        patchtst.prediction_length=128 \
        patchtst.patch_length=16 \
        patchtst.patch_stride=4 \
        patchtst.num_hidden_layers=8 \
        patchtst.num_attention_heads=8 \
        patchtst.d_model=512 \
        patchtst.quantizer_high=15.0 \
        patchtst.quantizer_low=-15.0 \
        patchtst.norm_type=rmsnorm \
        patchtst.channel_attention=true \
        patchtst.mode=predict \
        patchtst.pooling_type=mean \
        patchtst.loss=mse \
        patchtst.distribution_output=null \
        train.per_device_train_batch_size=128 \
        train.max_steps=400_000 \
        train.save_steps=100_000 \
        train.log_steps=1_000 \
        train.warmup_ratio=0.1 \
        train.torch_compile=true \
        train.weight_decay=1e-4 \
        quantizer.enabled=false \
        noiser.enabled=true \
        noiser.schedule_name=cosine \
        noiser.start=1.0 \
        noiser.end=0.0 \
        noiser.eps=0.008 \
        noiser.epoch_stop=0.5 \
        noiser.log_steps=100 \
        fixed_dim=4 \
        "$@"

# TODO: try playing with: patch_stride, pre_norm, pooling_type, dropout, head_dropout, attention_dropout, positional_dropout, ff_dropout, norm_type, mask_type



# # On multiple GPUs (example with 4 GPUs)
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
#         --nproc-per-node 4 \
#         scripts/patchtst/train.py \
#         shuffle_buffer_length=100_000 \
#         patchtst.pretrained_encoder_path=null \
#         patchtst.context_length=512 \
#         patchtst.prediction_length=128 \
#         patchtst.patch_length=16 \
#         patchtst.patch_stride=16 \
#         patchtst.num_hidden_layers=8 \
#         patchtst.num_attention_heads=8 \
#         patchtst.d_model=512 \
#         patchtst.quantizer_high=15.0 \
#         patchtst.quantizer_low=-15.0 \
#         patchtst.norm_type=rmsnorm \
#         patchtst.channel_attention=true \
#         patchtst.mode=predict \
#         train.per_device_train_batch_size=256 \
#         train.max_steps=300_000 \
#         train.save_steps=100_000 \
#         train.log_steps=1_000 \
#         train.warmup_ratio=0.1 \
#         train.torch_compile=true \
#         train.weight_decay=1e-4 \
#         quantizer.enabled=false \
#         noiser.enabled=false \
#         fixed_dim=3 \
#         "$@"
