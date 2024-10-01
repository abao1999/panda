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

We provide a script [dyst_data.py](scripts/dyst_data.py) for dataset generation. By default, this randomly splits all dysts into a 0.3 test/train split. You can manually specify your desired subset. Run `python scripts/dyst_data.py`. You can test the generated data and the data augmentations using our test scripts, e.g. `python tests/test_dyst_data.py Lorenz` and `python tests/test_augmentations.py Lorenz Rossler`.

We provide a script [skew_product.py](scripts/skew_product.py) to generate trajectories for pairs dynamical systems, where the first system (the driver) is driving the second system (the response). TODO: work in progress

### Testing Parameter and IC Perturbations
We provide a script [test_param_perturb.py](scripts/test_param_perturb.py) to test the effect of parameter perturbations on the generated trajectories from `dysts`. An example workflow:

```
./scripts/clean_dataset.sh train Lorenz
python tests/test_param_perturb.py Lorenz
python tests/test_dysts_data.py Lorenz
```

This will generate a trajectory ensemble for each parameter perturbation and save the trajectories to Arrow files.


## Training
Our train and eval scripts use hydra for hierarchical config, and you can log experiments to wandb. You can simply run `CUDA_VISIBLE_DEVICES=0 python scripts/train.py` to run with the default configs. To run with custom configs, 

```
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
            model_id=amazon/chronos-t5-small \
            run_name=finetune_1 \
            wandb.log=True \
            train.max_steps=1000 \
            train.save_steps=1000 \
            train.log_steps=50 \
```
(See `scripts/run_finetune.sh` for some example scripts). You can also run distributed training across multiple GPUs with `torchrun --nproc-per-node=6 scripts/train.py` (in this case, 6 GPUs). To run with custom configs,

```
torchrun --nproc-per-node=6 scripts/train.py \
        model_id=amazon/chronos-t5-large \
        n_tokens=4096 \
        context_length=512 \
        prediction_length=64 \
        run_name=finetune_large \
        wandb.log=True \
        wandb.group_name=finetune_large \
        train.max_steps=100_000 \
        train.save_steps=10_000 \
        train.log_steps=100 \
        train.per_device_train_batch_size=8 \
```

The torchrun launcher provides capability for distributed data-parallel (DDP) training. You can also try Huggingface's [accelerate](https://huggingface.co/docs/transformers/en/accelerate) launcher. For model-parallel training, see HF's [transformers model parallelism](https://huggingface.co/docs/transformers/v4.15.0/en/parallelism) guide.

We are fine-tuning [Chronos](https://github.com/amazon-science/chronos-forecasting) on trajectories from dynamical systems. Chronos is itself a variation of the [T5 architecture](https://huggingface.co/docs/transformers/en/model_doc/t5) fine-tuned on various benchmark univariate timeseries datasets. Specifically, Chronos fine-tunes an deep-narrow [efficient T5](https://huggingface.co/google/t5-efficient-large).

If you run into a weird miopen error, see: https://github.com/pytorch/pytorch/issues/60477

## Evaluation
To evalute the performance of a fine-tuned model, run `python scripts/evaluate.py` after setting the appropriate configuration in `configs/evaluation.yaml`. In particular, set `model_id` to point to the directory of your saved fine-tuned model checkpoint. The list of dynamical systems used for evaluation is also set in the configuration, but will default to using the test/train split.

## Experiments
+ Define a subset of 100 systems for training, randomly sample a dynamical system, make a trajectory from it, apply any augmentation, and then use that dataset for that batch. Set aside 30 attractors to test generalization. Can further augment training dataset with skew product systems.

## Ideas for Trained Models
+ Look at generalization to heldout dynamical systems
+ Probe for Koopman like internal dynamics
+ Error propagation to explain why long context beats precision limit of numerical integration
+ Encoder embeddings and token-level analysis
+ Ensure the fine-tuned model performance doesn't degrade on the timeseries it was previously trained on

## Development Goals
+ Add support for PatchTST
+ Add support for custom tokenizer and model architecture

## TODO
+ address this rocM warning: `libibverbs: Warning: couldn't load driver 'libmlx4-rdmav34.so': libmlx4-rdmav34.so: cannot open shared object file: No such file or directory`
+ Make dysts data under parameter and ic perturbations
+ Check attractor validity for parameter perturbations
+ Check attractor validity for skew system generation
+ For parameter perturbations and skew system generation, need to check that the system didn't bifurcate or diverge. Could just run an ADFuller stationarity test, as well as rule out constant and extremely large values. Make sure our heuristic tests covers all the bases.