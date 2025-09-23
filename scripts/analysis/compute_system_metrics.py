import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from dysts.analysis import max_lyapunov_exponent_rosenstein_multivariate
from dysts.base import DATAPATH_CONTINUOUS, DynSys
from tqdm import tqdm

from panda.utils.data_utils import load_trajectory_from_arrow


def system_iterator(eval_dirs: list[str]) -> Iterator[tuple[str]]:
    """Iterator for arrow file trajectories for multiple eval data directories

    Prevents opening too many file descriptors at once
    """
    for data_path in eval_dirs:
        for system_path in filter(lambda d: d.is_dir, Path(data_path).iterdir()):
            for traj_path in system_path.glob("*"):
                yield (system_path.name, traj_path)


def empirical_stiffness(traj: np.array, time_axis: int = -1) -> float:
    """Roughly measures the stiffness of a signal via the differences"""
    abs_diffs = np.abs(np.diff(traj, axis=time_axis))
    stiffness = abs_diffs.max() / abs_diffs.mean()
    return np.mean(stiffness, dtype=float)


@lru_cache
def get_period(system_name: str) -> float:
    systems = system_name.split("_")  # handle skew sytems
    return max(DynSys.load_system_metadata(sys, DATAPATH_CONTINUOUS)["period"] for sys in systems)


def metrics_worker(args: tuple) -> dict[str, float]:
    """Worker function for multiprocessing metrics computation"""
    system_name, traj_path = args

    trajectory, _ = load_trajectory_from_arrow(traj_path, one_dim_target=False)

    # hardcoded as all trajectories in the test were generated with period 40
    period = get_period(system_name)
    avg_dt, traj_len = period * 40 / trajectory.shape[1], 64
    max_lyap = max_lyapunov_exponent_rosenstein_multivariate(trajectory.T, tau=avg_dt, trajectory_len=traj_len)
    stiffness = empirical_stiffness(trajectory)

    metrics = {"max_lyap_r": max(max_lyap, 0), "stiffness": stiffness, "period": period, "avg_dt": avg_dt}
    return system_name, metrics


def compute_system_metrics(systems: Iterable[tuple[str]]) -> dict[str, list]:
    """Compute metrics in parallel

    NOTE: assumes order of metrics doesnt matter since we only care about the distribution of them
    """
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with Pool() as pool:
        for system_name, results in tqdm(
            pool.imap_unordered(metrics_worker, systems),
            desc="Processing systems",
        ):
            for metric, value in results.items():
                metrics[system_name][metric].append(value)

    return metrics


def metrics_dict_to_df(d: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    """Convert {system: {metric: [values]}} -> wide DataFrame with one row per sample.

    Columns: 'system', optional 'sample', then one column per metric name.
    """
    frames: list[pd.DataFrame] = []
    for system, metrics in d.items():
        sys_frame = pd.DataFrame({metric: pd.Series(values, dtype=float) for metric, values in metrics.items()})
        sys_frame.insert(0, "system", system)
        frames.append(sys_frame)

    return pd.concat(frames, ignore_index=True)


def main():
    work_dir = os.environ.get("WORK", "/stor/work/AMDG_Gilpin_Summer2024")
    eval_dirs = [
        f"{work_dir}/data/improved/final_base40/test_zeroshot",
        f"{work_dir}/data/improved/final_base40/test_zeroshot_z5_z10",
        f"{work_dir}/data/improved/final_skew40/test_zeroshot",
        f"{work_dir}/data/improved/final_skew40/test_zeroshot_z5_z10",
    ]
    trajectories = system_iterator(eval_dirs)
    metrics = compute_system_metrics(trajectories)

    df = metrics_dict_to_df(metrics)
    df.to_csv("data/system_metrics.csv", index=False)


if __name__ == "__main__":
    main()
