# chronos-dysts
Chronos fine-tuned on Dynamical Systems

## Setup
Install [dysts](https://github.com/williamgilpin/dysts) for dynamical systems with `pip install --no-deps git+https://github.com/williamgilpin/dysts`

To setup, run:
```
$ pip install -r requirements.txt
$ pip install -e .
```

## Generating Dyst Dataset
We structure our Arrow files as multivariate trajectories saved per sample instance, with default `1024` numerical integration timesteps. Each sample instance is a specific parameter perturbation and initial condition. For example, each Arrow file `[sample_idx]_T-1024.arrow` within `data/train/Lorenz` corresponds to a single sample instance of the Lorenz system, and contains trajectories for all coordinates (dimensions) i.e. a numpy array of shape (3, 1024) for the Lorenz system.  

We provide a script `scripts/dyst_data.py` for dataset generation. By default, this randomly splits all dysts into a 0.3 test/train split. You can manually specify your desired subset. We also provide several on-the-fly dataset augmentations compatible with the GluonTS framework, in `chronos_dysts/augmentations.py`.

Currently implemented data augmentations:
- RandomAffineTransform
- RandomConvexCombinationTransform
- RandomProjectedSkewTransform

Skew product systems are a work in progress.

You can test the generated data and the data augmentations using our test scripts, e.g. `python tests/test_dyst_data.py Lorenz` and `python tests/test_augmentations.py Lorenz Rossler`.

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
(See `scripts/run_finetune.sh` for some example scripts)

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

## TODO
+ Validate the current augmentation strategies
+ For parameter perturbations and skew system generation, need to check that the system didn't bifurcate or diverge. Could just run an ADFuller stationarity test, as well as rule out constant and extremely large values. Make sure our heuristic tests covers all the bases.
+ Custom tokenizer and model architecture