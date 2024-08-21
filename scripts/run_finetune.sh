# On single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train.py run_name=finetune_1 wandb.log=True

# # On multiple GPUs (example with 6 GPUs)
# torchrun --nproc-per-node=6 scripts/training/train.py

# accelerate launch scripts/training/train.py
