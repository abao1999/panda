import logging
import time
from dataclasses import dataclass

import numpy as np
from dysts.base import BaseDyn

logger = logging.getLogger(__name__)


@dataclass
class TimeLimitEvent:
    """
    Event to check if integration is taking too long
    """

    system: BaseDyn
    max_duration: float
    terminal: bool = True
    verbose: bool = False

    def __post_init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def __call__(self, t, y):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration:
            if self.verbose:
                logger.warning(
                    f"{self.system.name} exceeded time limit: {elapsed_time:.2f}s"
                )
            return 0
        return 1


@dataclass
class InstabilityEvent:
    """
    Event to detect instability during numerical integration

    Ignores unbounded indices from the system
    """

    system: BaseDyn
    threshold: float
    terminal: bool = True
    verbose: bool = False

    def __call__(self, t, y):
        bounded_coords = np.abs(np.delete(y, self.system.unbounded_indices))
        if np.any(bounded_coords > self.threshold) or np.any(np.isnan(y)):
            if self.verbose:
                logger.warning(
                    f"Instability in {self.system.name} @ t={t:.3f}: {np.abs(y).max():.3f}"
                )
            return 0
        return 1


@dataclass
class TimeStepEvent:
    """
    Event to check if the system time step is invalid
    """

    system: BaseDyn
    last_t: float = float("inf")
    min_step: float = 1e-20
    terminal: bool = True
    verbose: bool = False

    def __call__(self, t, y):
        t_diff = abs(t - self.last_t)
        if t_diff < self.min_step:
            if self.verbose:
                logger.warning(f"{self.system.name} time step too small: {t_diff:.2e}")
            return 0

        self.last_t = t
        return 1
