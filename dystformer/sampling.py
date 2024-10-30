import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from dysts.base import BaseDyn
from dysts.sampling import BaseSampler
from numpy.typing import NDArray

Array = NDArray[np.float64]


# Event functions for solve_ivp
@dataclass
class TimeLimitEvent:
    """
    Event to check if integration is taking too long
    """

    max_duration: float
    terminal: bool = True

    def __post_init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def __call__(self, t, y):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration:
            print("Integration stopped due to time limit.")
            return 0  # Trigger the event
        return 1  # Continue the integration


@dataclass
class InstabilityEvent:
    """
    Event to detect instability during numerical integration
    """

    threshold: float
    terminal: bool = True

    def __call__(self, t, y):
        if np.any(np.abs(y) > self.threshold):
            print("y: ", y)
            print("Integration stopped due to instability.")
            return 0  # Trigger the event
        return 1  # Continue the integration


@dataclass
class SignedGaussianParamSampler(BaseSampler):
    """Sample gaussian perturbations for system parameters

    NOTE:
        - This is a MWE of a parameter transform
        - Other parameter transforms should follow this dataclass template

    Args:
        scale: std (isotropic) of gaussian used for sampling
        sign_match_probability: proportion of perturbations that match the sign of the parameter
    """

    scale: float = 1e-2
    sign_match_probability: float = 0.0
    verbose: bool = False

    def __call__(
        self, name: str, param: Array, system: Optional[BaseDyn] = None
    ) -> Array | float:
        # scale each parameter relatively
        shape = 1 if np.isscalar(param) else param.shape

        # avoid shape errors
        flat_param = np.array(param, dtype=np.float32).flatten()
        scale = np.abs(flat_param) * self.scale
        cov = np.diag(np.square(scale))
        perturbation = self.rng.multivariate_normal(
            mean=np.zeros_like(flat_param), cov=cov
        ).reshape(shape)

        # demote singletons to scalar
        if np.isscalar(param):
            perturbation = perturbation.item()

        if self.rng.random() < self.sign_match_probability:
            perturbation = np.sign(param) * np.abs(perturbation)

        perturbed_param = param + perturbation

        if self.verbose:
            if system is not None:
                print(f"System: {system.name}")
            print(f"Parameter name: {name}")
            print(f"--> Original parameter: {param}")
            print(f"--> Perturbed parameter: {perturbed_param}")

        return perturbed_param


@dataclass
class OnAttractorInitCondSampler(BaseSampler):
    """
    Sample points from the attractor of a system

    Subtleties:
        - This is slow, it requires integrating each system with its default
          parameters before sampling from the attractor.
        - The sampled initial conditions from this sampler are necessarily
          tied to the attractor defined by the default parameters.

    Args:
        reference_traj_length: Length of the reference trajectory to use for sampling ic on attractor.
        reference_traj_transient: Transient length to ignore for the reference trajectory
        trajectory_cache: Cache of reference trajectories for each system.
        events: integration events to pass to solve_ivp
    """

    reference_traj_length: int = 4096
    reference_traj_transient: float = 0.2
    trajectory_cache: Dict[str, Array] = field(default_factory=dict)
    verbose: bool = False
    events: Optional[List[Callable]] = None
    recompute_standardization: bool = False

    def __post_init__(self):
        # super().__post_init__()
        assert (
            0 < self.reference_traj_transient < 1
        ), "Transient must be a fraction of the trajectory length"
        self.transient = int(self.reference_traj_length * self.reference_traj_transient)

    def __call__(self, ic: Array, system: BaseDyn) -> Array:
        if system.name is None:
            raise ValueError("System must have a name")

        # make reference trajectory if not already cached
        if system.name not in self.trajectory_cache:
            # Integrate the system with default parameters
            reference_traj = system.make_trajectory(
                self.reference_traj_length,
                events=self.events,
                standardize=False,
            )

            # if integrate fails, resulting in an incomplete trajectory
            if reference_traj is None:
                warnings.warn(
                    f"Failed to integrate the system {system.name} with ic {system.ic} and params {system.params}"
                )
                return ic

            # renormalize with respect to reference trajectory
            # this should work since system is passed by reference
            if self.recompute_standardization:
                system.mean = reference_traj.mean(axis=0)
                system.std = reference_traj.std(axis=0)

            reference_traj = reference_traj[self.transient :]
            self.trajectory_cache[system.name] = reference_traj

        trajectory = self.trajectory_cache[system.name]

        # Sample a new initial condition from the attractor
        new_ic = self.rng.choice(trajectory)

        if self.verbose:
            print(f"System: {system.name}")
            print(f"--> Original initial condition: {ic}")
            print(f"--> New initial condition: {new_ic}")

        return new_ic


@dataclass
class GaussianInitialConditionSampler(BaseSampler):
    """
    Sample gaussian perturbations for each initial condition in a given system list
    """

    scale: float = 1e-4
    verbose: bool = False  # for testing purposes

    def __call__(self, ic: Array, system: BaseDyn) -> Array:
        """
        Sample a new initial condition from a multivariate isotropic Gaussian.

        Args:
            ic (Array): The current initial condition.

        Returns:
            Array: A resampled version of the initial condition.
        """
        # Scale the covariance relative to each dimension
        scaled_cov = np.diag(np.square(ic * self.scale))
        perturbed_ic = self.rng.multivariate_normal(mean=ic, cov=scaled_cov)

        if self.verbose:
            print(f"System: {system.name}")
            print(f"--> Original initial condition: {ic}")
            print(f"--> Perturbed initial condition: {perturbed_ic}")

        return perturbed_ic
