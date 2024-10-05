# CUDA_VISIBLE_DEVICES=0 \
# python scripts/patchtst/train.py \
#         run_name=patchtst_lorenz_overfit \
#         patchtst.context_length=512 \
#         patchtst.prediction_length=64 \
#         patchtst.num_input_channels=3 \
#         shuffle_buffer_length=1000 \
#         wandb.log=False \
#         wandb.group_name=finetune_large \
#         train.max_steps=20_000 \
#         train.save_steps=10_000 \
#         train.log_steps=100 \

# On multiple GPUs (example with 6 GPUs)
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
        --nproc-per-node 4 \
        scripts/patchtst/train.py \
        run_name=patchtst_lorenz_overfit \
        shuffle_buffer_length=100_000 \
        patchtst.context_length=512 \
        patchtst.prediction_length=64 \
        patchtst.patch_length=16 \
        patchtst.patch_stride=16 \
        patchtst.num_hidden_layers=8 \
        patchtst.num_attention_heads=8 \
        patchtst.d_model=128 \
        train.per_device_train_batch_size=256 \
        train.max_steps=300_000 \
        train.save_steps=100_000 \
        train.log_steps=1_000 \
        train.torch_compile=false \
        "$@"
