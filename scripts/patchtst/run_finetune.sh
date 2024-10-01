# Debug train script
# On single GPU
CUDA_VISIBLE_DEVICES=0 \
python scripts/patchtst/train.py \
        run_name=patchtst_lorenz_overfit \
        patchtst.context_length=512 \
        patchtst.prediction_length=64 \
        patchtst.num_input_channels=3 \
        shuffle_buffer_length=1000 \
        wandb.log=False \
        wandb.group_name=finetune_large \
        train.max_steps=20_000 \
        train.save_steps=10_000 \
        train.log_steps=100 \

# # On multiple GPUs (example with 6 GPUs)
# torchrun --nproc-per-node=6 scripts/patchtst/train.py \
#         run_name=patchtst_lorenz_overfit \
#         patchtst.context_length=512 \
#         patchtst.prediction_length=64 \
#         patchtst.num_input_channels=3 \
#         shuffle_buffer_length=1000 \
#         wandb.log=True \
#         wandb.group_name=finetune_large \
#         train.max_steps=20_000 \
#         train.save_steps=10_000 \
#         train.log_steps=100 \

# accelerate launch scripts/training/train.py