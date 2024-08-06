# # Fine-tune `amazon/chronos-t5-small` for 100 steps with initial learning rate of 1e-3

# CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py --config scripts/training/configs/chronos-t5-small.yaml \
#     --model-id amazon/chronos-t5-small \
#     --no-random-init \
#     --max-steps 100 \
#     --learning-rate 0.001

# # On multiple GPUs (example with 6 GPUs)
# torchrun --nproc-per-node=6 scripts/training/train.py --config scripts/training/configs/chronos-t5-small.yaml

# On single GPU
CUDA_VISIBLE_DEVICES=0 python scripts/training/train.py --config scripts/training/configs/chronos-t5-small.yaml

# accelerate launch scripts/training/train.py --config scripts/training/configs/chronos-t5-small.yaml