# dystformer
PatchTST trained on Dynamical Systems with channel mixing

Baselines:
+ PatchTST trained on Dynamical Systems without channel mixing
+ Chronos fine-tuned on Dynamical Systems
+ Zero-Shot Models

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

## Generating Dyst Dataset
We structure our Arrow files as multivariate trajectories saved per sample instance, with default `1024` numerical integration timesteps. Each sample instance is a specific parameter perturbation and initial condition. For example, each Arrow file `[SAMPLE_IDX]_T-1024.arrow` within `[DATA_DIR]/train/Lorenz` corresponds to a single sample instance of the Lorenz system, and contains trajectories for all coordinates (dimensions) i.e. a numpy array of shape (3, 1024) for the Lorenz system.  

We provide several on-the-fly dataset augmentations compatible with the GluonTS framework, in [dystformer/augmentations.py](dystformer.augmentations). By default, all of these augmentations are used during training, but the choice of augmentations can be directly specified in the [dataset config](config/dataset.yaml).

Currently implemented data augmentations:
- RandomAffineTransform
- RandomConvexCombinationTransform
- RandomProjectedSkewTransform

We provide a script [make_dyst_data.py](scripts/make_dyst_data.py) for dataset generation. By default, this randomly splits all dysts into a 0.3 test/train split. You can manually specify your desired subset. Running `python scripts/make_dyst_data.py` will save trajectories for systems in the train and test splits to their respective data sub-directories. You can test the generated data and the data augmentations using our test scripts, e.g. `python tests/test_saved_data.py Lorenz` and `python tests/test_augmentations.py Lorenz Rossler`. Note that the `--one_dim_target` flag is required for Chronos dataset format, where the coordinates are split and then saved. An example workflow:

```
python tests/make_dyst_data.py
python tests/test_saved_data.py all --split train
python tests/test_saved_data.py all --split test
python tests/test_augmentations.py Lorenz Rossler --split train
```

We provide a script [make_skew_systems.py](scripts/make_skew_systems.py) to generate trajectories for pairs dynamical systems, where the first system (the driver) is driving the second system (the response). TODO: work in progress. Example usage: `python scripts/make_skew_systems.py Blasius Aizawa --couple_phase_space True --couple_flows False`

### Testing Parameter and IC Perturbations
We provide a script [test_attractor.py](scripts/test_attractor.py) to test parameter perturbations and check if the generated trajectories are valid attractors. An example workflow:

```
./scripts/clean_dataset.sh train all
python tests/test_attractor.py Lorenz
python tests/test_saved_data.py Lorenz
```

To investigate all failed attractors, run:
```
python tests/test_attractor.py all
python tests/test_saved_data.py all --split failed_attractors --metadata_path tests/attractor_checks.json --samples_subset failed_samples
```

This will generate a trajectory ensemble for each parameter perturbation and save the trajectories to Arrow files.

## Training
Our train and eval scripts use hydra for hierarchical config, and you can log experiments to wandb. You can simply run `CUDA_VISIBLE_DEVICES=0 python scripts/patchtst/train.py` or `CUDA_VISIBLE_DEVICES=0 python scripts/chronos/train.py` to run with the default configs. To run with custom configs, 

### PatchTST Training
See `scripts/patchtst/run_finetune.sh` for an example script.

### Chronos Training
We are fine-tuning [Chronos](https://github.com/amazon-science/chronos-forecasting) on trajectories from dynamical systems. Chronos is itself a variation of the [T5 architecture](https://huggingface.co/docs/transformers/en/model_doc/t5) fine-tuned on various benchmark univariate timeseries datasets. Specifically, Chronos fine-tunes an deep-narrow [efficient T5](https://huggingface.co/google/t5-efficient-large).

See `scripts/run_finetune.sh` for some example scripts. 

### Distributed Training
The torchrun launcher provides capability for distributed data-parallel (DDP) training. You can also try Huggingface's [accelerate](https://huggingface.co/docs/transformers/en/accelerate) launcher. For model-parallel training, see HF's [transformers model parallelism](https://huggingface.co/docs/transformers/v4.15.0/en/parallelism) guide.

If you run into a weird miopen error, see: https://github.com/pytorch/pytorch/issues/60477

## Evaluation
To evalute the performance of a fine-tuned model, run `python scripts/evaluate.py` after setting the appropriate configuration in `configs/evaluation.yaml`. In particular, set `model_id` to point to the directory of your saved fine-tuned model checkpoint. The list of dynamical systems used for evaluation is also set in the configuration, but will default to using the test/train split.

## Notes
+ does it make more sense to reestimate period - a surrogate for the timescale, or the lyapunov exponent - a well defined quantity? What about using first UPO?
+ Must decide whether we want to use MLM or causal prediction for pretraining
        - MLM is what patchtst does
        - causal prediction is what chronos does
+ look into different loss functions for the pretraining
+ one possible route to the jagged dim problem: for each batch of traj data, store ordered list of each trajs phase space dimension, flatten the batch dimensions along with the phase space dimension i.e. -> [batch_size * num_dimensions, context_length + prediction_length], then in forward decode using the stored dimensions

## Development Goals

Please grade each item with the convention [Priority | Difficulty], where priority can be high, medium, or low, and difficulty can be high, medium, or low.

+ [HIGH | EASY] implement correct procedure for instance normalization + regenerate standardized training data
+ [HIGH | HARD] limit cycle test for generating data
+ [HIGH | EASY] are random or regular predictions windows better?
+ [MEDIUM | MEDIUM] implement patchtst for causal prediction pretraining
+ [MEDIUM | MEDIUM] utilize the `probabilities` variable (see `dystformer/scripts/patchtst/train.py`) for the dataset to equalize dysts sampling according to the distribution of phase space dimension. How to weigh the augmentations in the context of dimension distribution (dimensions can be arbitrary)?
+ [MEDIUM | MEDIUM] implement a better parameter sampler for generating dyst data
+ [LOW | MEDIUM] different positional encoding might allow for torch compile to work 
+ [LOW | EASY] get to the bottom of this strange miopen fix https://github.com/pytorch/pytorch/issues/60477#issuecomment-1574453494
+ [LOW | HARD] add flash attention support for AMD, see this warning: 
```
[W sdp_utils.cpp:264] Warning: 1Torch was not compiled with flash attention. (function operator())
[W sdp_utils.cpp:320] Warning: 1Torch was not compiled with memory efficient attention. (function operator())
```
+ [LOW | EASY] Plotting utils to make it easy to debug computed quantities used in attractor checks e.g. power spectrum, polynomial fits, etc. And, enable multiprocessed attractor tests across ensemble