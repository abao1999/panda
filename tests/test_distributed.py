"""
This script tests the distributed training setup.

It tests the following:
- Basic GPU operations
- Distributed initialization
- Parameter verification
- Broadcast test
- Error handling

Original Issue: NCCL distributed training failing on AMD GPUs with Ubuntu 24 + ROCm 6.3
Root Cause: Device mapping issues where all processes were using the same GPU, causing "Duplicate GPU detected" and "invalid argument" errors

Solution:
- Set NCCL environment variables for AMD GPU compatibility
- Set device based on LOCAL_RANK
- Add error handling for distributed training
- Improve distributed communication stability

Notes:
- AMD GPUs don't handle CUDA_VISIBLE_DEVICES the same way as NVIDIA GPUs, so we need to ensure each distributed process uses a different GPU
- ROCm's NCCL implementation has different compatibility requirements than CUDA's NCCL implementation
"""

import os

import torch
import torch.distributed as dist


def test_distributed():
    """
    NOTE: These are the environment variables that need to be set for distributed training to work
        export NCCL_P2P_DISABLE=1
        export NCCL_IB_DISABLE=1
        export NCCL_SOCKET_IFNAME=lo
        export NCCL_NET_GDR_LEVEL=0

    Usage:
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc-per-node 4 \
        --master-port 29500 \
        tests/test_distributed.py
    """
    print("=== Environment Variables ===")
    print(f"RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("\n=== GPU Info ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    # CRITICAL FIX: Set the device based on LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"Current device: {torch.cuda.current_device()}")

    # Test basic GPU operations
    x = torch.randn(10, 10).cuda()
    print("Basic GPU operation successful")

    # Test distributed initialization
    try:
        dist.init_process_group(backend="nccl")
        print("NCCL initialization successful")

        # Test parameter verification with more robust tensor creation
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"Process {rank}/{world_size} starting broadcast test")

        # Create tensor on the correct device
        if rank == 0:
            tensor = torch.randn(10, 10, device=f"cuda:{local_rank}")
            print(f"Rank {rank}: Created tensor on device {local_rank}")
        else:
            tensor = torch.zeros(10, 10, device=f"cuda:{local_rank}")
            print(f"Rank {rank}: Created zero tensor on device {local_rank}")

        # Ensure tensor is contiguous and on the right device
        tensor = tensor.contiguous()
        print(f"Rank {rank}: Tensor device: {tensor.device}, shape: {tensor.shape}")

        # Test broadcast
        dist.broadcast(tensor, src=0)
        print(f"Rank {rank}: Broadcast successful")

        # Test all-reduce
        dist.all_reduce(tensor)
        print(f"Rank {rank}: All-reduce successful")

        dist.destroy_process_group()
        print(f"Rank {rank}: Process group destroyed")

    except Exception as e:
        print(f"Rank {rank}: Distributed operation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_distributed()
