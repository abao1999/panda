## Debug train script
# # On single GPU
# CUDA_VISIBLE_DEVICES=0 \
# python scripts/train.py \
#             run_name=finetune_large \
#             wandb.log=True \
#             train.max_steps=100_000 \
#             train.save_steps=10_000 \
#             train.log_steps=100 \

# On multiple GPUs (example with 6 GPUs)
torchrun --nproc-per-node=6 scripts/train.py \
        run_name=finetune_large \
        wandb.log=True \
        wandb.group_name=finetune_large \
        train.max_steps=100_000 \
        train.save_steps=10_000 \
        train.log_steps=100 \

# accelerate launch scripts/training/train.py
