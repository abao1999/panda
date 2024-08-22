## Debug train script
# On single GPU
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
            run_name=finetune_1 \
            wandb.log=True \
            train.max_steps=100 \
            train.save_steps=100 \
            train.log_steps=10 \

# # On multiple GPUs (example with 6 GPUs)
# torchrun --nproc-per-node=6 scripts/training/train.py

# accelerate launch scripts/training/train.py
