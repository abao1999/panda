# Debug train script
# # On single GPU
CUDA_VISIBLE_DEVICES=0 \
python scripts/chronos/train.py \
        run_name=finetune_large \
        wandb.log=false \
        train.max_steps=100_000 \
        train.save_steps=10_000 \
        train.log_steps=100 \
        train_data_dir=$WORK/data/chronos_train \

# On multiple GPUs (example with 6 GPUs)
# torchrun --nproc-per-node=3 scripts/chronos/train.py \
#         run_name=finetune_small \
#         wandb.log=false \
#         wandb.group_name=finetune_large \
#         train.max_steps=100_000 \
#         train.save_steps=10_000 \
#         train.log_steps=100 \
#         train_data_dir=$WORK/data/chronos_train \

# accelerate launch scripts/training/train.py
