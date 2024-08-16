# chronos-dysts
Chronos fine-tuned on Dynamical Systems

## Setup
Install [chronos](https://github.com/amazon-science/chronos-forecasting) with the optional training and evaluation dependencies. Also install [dysts](https://github.com/williamgilpin/dysts) for dynamical systems.

To setup, run:
```
$ pip install -r requirements.txt
$ pip install -e .
```

## Experiments
+ Define a subset of 100 systems for training, randomly sample a dynamical system, make a trajectory from it, apply any augmentation, and then use that dataset for that batch. Set aside 30 attractors to test generalization.

## Ideas for Trained Models
+ Probe for Koopman like internal dynamics
+ Error propagation to explain why long context beats precision limit of numerical integration

## TODO
+ Think of some "physics based" data augmentation strategies. Random affine transforms of time series are still valid dynamical systems, as well as random transcendental functions.
+ Functionality to add random values to the parameters of each attractor, while checking that the system didn't bifurcate or diverge. Could just run an ADFuller stationarity test, as well as rule out constant and extremely large values
+ Make model configs more efficient i.e. hydra config, since all the model yaml files repeat each other except for model name.