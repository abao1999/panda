# panda
Panda: Patched Attention for Nonlinear Dynamics

This repository contains the code to reproduce the experiments presented in our arXiv preprint [arXiv:2505.13755](https://arxiv.org/abs/2505.13755)

We have released model weights on HF at https://huggingface.co/GilpinLab/panda

We are also in the process of scaling up our training and model size, so stay tuned!

## Setup
Install the most up-to-date version of [dysts](https://github.com/williamgilpin/dysts) for dynamical systems with `pip install --no-deps git+https://github.com/williamgilpin/dysts`. Consider installing `numba` for faster numerical integration.

To setup, run:
```
$ pip install -e .
```

If training on AMD GPUs, install with the ROCm extras:
```
$ pip install -e .[rocm] --extra-index-url https://download.pytorch.org/whl/rocm5.7
```

## Dataset Generation

## Our Model

## Training Our Model

## Training Baselines

## Evaluation

## Citation
If you use this codebase or otherwise find our work valuable, please cite us:
```
@misc{lai2025panda,
      title={Panda: A pretrained forecast model for universal representation of chaotic dynamics}, 
      author={Jeffrey Lai and Anthony Bao and William Gilpin},
      year={2025},
      eprint={2505.13755},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.13755}, 
}
```
