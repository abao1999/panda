import os

from dystformer.dyst_data import DystData
from dystformer.sampling import (
    InstabilityEvent,
    TimeLimitEvent,
)
from dystformer.utils import (
    split_systems,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")
DELAY_SYSTEMS = [
    "MackeyGlass",
    "IkedaDelay",
    "SprottDelay",
    "VossDelay",
    "ScrollDelay",
    "PiecewiseCircuit",
]

if __name__ == "__main__":
    # set random seed
    rseed = 999  # we are using same seed for split and ic and param samplers

    # generate split of dynamical systems
    test, train = split_systems(0.3, seed=rseed, excluded_systems=DELAY_SYSTEMS)

    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60 * 2)  # 2 min time limit
    instability_event = InstabilityEvent(threshold=1e4)

    dyst_data_generator = DystData(
        rseed=rseed,
        num_periods=5,
        num_points=1024,
        num_ics=5,  # only activates ic sampler if > 1
        num_param_perturbations=1,  # only activates param sampler if > 1
        events=[time_limit_event, instability_event],
        verbose=True,
    )

    # make the train split
    dyst_data_generator.save_dyst_ensemble(
        dysts_names=train,
        split="train",
        samples_save_interval=1,
        save_dir=DATA_DIR,
    )

    # make the test split
    dyst_data_generator.save_dyst_ensemble(
        dysts_names=test,
        split="test",
        samples_save_interval=1,
        save_dir=DATA_DIR,
    )
