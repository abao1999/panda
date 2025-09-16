# panda
Panda: Patched Attention for Nonlinear Dynamics

This repository contains the code to reproduce the experiments presented in our arXiv preprint [arXiv:2505.13755](https://arxiv.org/abs/2505.13755)

We have released model weights on HF at https://huggingface.co/GilpinLab/panda

We also have MLM weights up on HF at https://huggingface.co/GilpinLab/panda_mlm

We have also released datasets on HF. 

- Dataset for our scaled-up training (checkpoints coming soon): https://huggingface.co/datasets/GilpinLab/skew-mixedp-ic16
- Dataset for scaling law experiment: https://huggingface.co/datasets/GilpinLab/scalinglaw
- Original training dataset https://huggingface.co/datasets/GilpinLab/skew40.

**We are in the process of scaling up our training and model size. We are also working on a larger dataset with improved filtering for chaotic behavior. Stay tuned!**

Paper abstract:

>"Chaotic systems are intrinsically sensitive to small errors, challenging efforts to construct predictive data-driven models of real-world dynamical systems such as fluid flows or neuronal activity.
Prior efforts comprise either specialized models trained separately on individual time series, or foundation models trained on vast time series databases with little underlying dynamical structure.
Motivated by dynamical systems theory, we present *Panda*, *P*atched *A*ttention for *N*onlinear *D*yn*A*mics.
We train *Panda* on a novel synthetic, extensible dataset of $2 \times 10^4$ chaotic dynamical systems that we discover using an evolutionary algorithm.
Trained purely on simulated data, *Panda* exhibits emergent properties: zero-shot forecasting of unseen real world chaotic systems, and nonlinear resonance patterns in cross-channel attention heads.
Despite having been trained only on low-dimensional ordinary differential equations, *Panda* spontaneously develops the ability to predict partial differential equations without retraining.
We demonstrate a neural scaling law for differential equations, underscoring the potential of pretrained models for probing abstract mathematical domains like nonlinear dynamics."

## Setup
**We have merged much of our dataset generation functionality into the dysts repo. See dysts for more up-to-date development.**

*NOTE:* When cloning this repo, to avoid downloading the large commit history (~ 60 MB) we recommend a *shallow clone*:

`git clone --depth=1 git@github.com:abao1999/panda.git`

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to install the dependencies. After cloning, create a virtual environment:
```
uv venv --python 3.13
source .venv/bin/activate
```
And simply run `uv sync` to install dependencies. To install torch for GPU run:
```
uv sync --extra-index-url https://download.pytorch.org/whl/rocm6.3
```
To install torch with ROCm, or for CUDA:
```
uv sync --extra-index-url https://download.pytorch.org/whl/cuda12.9
```

For the development setup, use:
```
uv sync --group dev
```

## Dataset Generation
Our dataset consists of parameter perturbations of base and skew systems. Each trajectory is a numerically integrated system of coupled ODEs that we filter according to the methodology outlined in our preprint. To run the data generation, see our scripts for [making trajectories from saved params](https://github.com/abao1999/panda/blob/main/scripts/make_dataset_from_params.py), [parameter perturbations of skew systems](https://github.com/abao1999/panda/blob/main/scripts/make_skew_systems.py), and [parameter perturbations of base systems](https://github.com/abao1999/panda/blob/main/scripts/make_dyst_data.py). For ease of use we have also provided an example data generation [bash script](https://github.com/abao1999/panda/blob/main/scripts/bash_scripts/run_data_generation.sh) that calls these scripts.

## Our Model
![model schematic](data/model_schematic.png)

## Training Our Model
We provide example bash scripts to train our model, both for [forecasting](https://github.com/abao1999/panda/blob/main/scripts/patchtst/run_predict_finetune.sh) and for [MLM](https://github.com/abao1999/panda/blob/main/scripts/patchtst/run_pretrain.sh) (completions). Recall that it is possible to train an MLM checkpoint and use the encoder for prediction finetuning (SFT) for forecasting. See our [training script](https://github.com/abao1999/panda/blob/main/scripts/patchtst/train.py) for more details.

## Evaluation
In [notebooks/load_model_from_hf.ipynb](https://github.com/abao1999/panda/blob/main/notebooks/load_model_from_hf.ipynb) we provide a minimal working example of loading our trained checkpoint from HuggingFace and running inference (generating forecasts). For reproducibility, we also provide a serialized [json file](https://github.com/abao1999/panda/blob/main/data/params_test_zeroshot/filtered_params_dict.json) (~ 10 MB) containing the parameters for some of our held-out skew systems. These parameters can then be loaded and used to generate trajectories from the corresponding systems.

For a more thorough evaluation, see our [evaluation script](https://github.com/abao1999/panda/blob/main/scripts/patchtst/evaluate.py), which we used to present the results in our preprint. A corresponding script exists for each of the baselines we evaluate on, within `scripts`.

In [notebooks/load_mlm_from_hf.ipynb](https://github.com/abao1999/panda/blob/main/notebooks/load_mlm_from_hf.ipynb) we provide a minimal working example of loading our trained MLM checkpoint from HuggingFace and generating completions.

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
