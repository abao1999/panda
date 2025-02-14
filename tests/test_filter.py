import os
import random
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from gluonts.dataset.common import FileDataset

from dystformer.utils import plot_univariate_trajs, safe_standardize


def load_dataset(path: Path, num_samples: int) -> dict[str, list[FileDataset]]:
    data_dict = {}
    data_dir = os.path.expandvars(path)
    system_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    for system_dir in random.sample(system_dirs, num_samples):
        system_name = system_dir.name
        system_files = list(system_dir.glob("*"))
        data_dict[system_name] = [
            next(
                iter(FileDataset(path=Path(file_path), freq="h", one_dim_target=False))
            )["target"]
            for file_path in system_files
            if file_path.is_file()
        ]
    return data_dict


def compute_autocorrelation(data: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation up to max_lag for a time series."""
    mean = np.mean(data)
    var = np.var(data)
    normalized_data = data - mean
    acf = np.correlate(normalized_data, normalized_data, mode="full")
    acf = acf[len(acf) // 2 : len(acf) // 2 + max_lag]
    return acf / (var * len(data))


def is_smooth(
    time_series: np.ndarray, decay_threshold: float = 0.1, window_size: int = 10
) -> bool:
    """Test if a time series is smooth using autocorrelation decay rate.

    Args:
        time_series: Input signal of shape (features, time_steps)
        decay_threshold: Maximum allowed average decay rate per lag
        window_size: Number of lags to analyze for decay rate

    Returns:
        bool: True if signal is considered smooth
    """
    normalized = safe_standardize(time_series)
    is_smooth = True
    for i in range(normalized.shape[0]):
        acf = compute_autocorrelation(normalized[i])

        # Compute average decay rate over the first window_size lags
        # Smooth signals should have small (slow) decay rates
        decay_rates = (acf[:-1] - acf[1:])[:window_size]
        avg_decay_rate = np.mean(decay_rates)
        is_smooth &= bool(avg_decay_rate < decay_threshold)

    return is_smooth


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    systems = load_dataset(cfg.eval.data_path, cfg.eval.num_systems)
    systems_mask = defaultdict(list)
    for system_name, time_series in systems.items():
        for ts in time_series:
            systems_mask[system_name].append(is_smooth(ts.T))

    num_nonsmooth = {sys: len(mask) - sum(mask) for sys, mask in systems_mask.items()}
    print(f"Number of nonsmooth systems: {num_nonsmooth}")

    nonsmooth_systems = {
        system_name: [
            time_series
            for time_series, is_smooth in zip(
                systems[system_name], systems_mask[system_name]
            )
            if not is_smooth
        ]
        for system_name in systems
    }
    filtered_nonsmooth_systems = {
        system_name: np.stack(time_series, axis=0)
        for system_name, time_series in nonsmooth_systems.items()
        if len(time_series) > 0
    }

    plot_univariate_trajs(
        filtered_nonsmooth_systems,
        save_path="tests/figs/filtering",
        figsize=(12, 8),
        max_samples=3,
    )


if __name__ == "__main__":
    main()
