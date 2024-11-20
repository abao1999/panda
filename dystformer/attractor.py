"""
Suite of tests to determine if generated trajectories are valid attractors
"""

import functools
import inspect
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from dysts.analysis import max_lyapunov_exponent_rosenstein
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


@dataclass
class AttractorValidator:
    """
    Framework to add tests, which are executed sequentially to determine if generated trajectories are valid attractors.
    Upon first failure, the trajectory sample is added to the failed ensemble.
    To add custom tests, define functions that take a trajectory and return a boolean (True if the trajectory passes the test, False otherwise).
    """

    transient_time_frac: float = 0.05  # should be low, should be on attractor
    plot_save_dir: Optional[str] = None
    verbose: bool = False
    tests: Optional[List[Callable]] = None

    def __post_init__(self):
        self.failed_checks = defaultdict(list)  # Dict[str, List[Tuple[int, str]]]
        self.valid_dyst_counts = defaultdict(int)  # Dict[str, int]
        self.failed_samples = defaultdict(list)  # Dict[str, List[int]]
        self.valid_samples = defaultdict(list)  # Dict[str, List[int]]
        self.tests = self.tests or []

    def reset(self):
        """
        Reset all defaultdict attributes to their initial state.
        """
        self.failed_checks.clear()
        self.valid_dyst_counts.clear()
        self.failed_samples.clear()
        self.valid_samples.clear()

    def _execute_test_fn(
        self,
        test_fn: Callable,
        dyst_name: str,
        traj_sample: np.ndarray,
        sample_idx: int,
    ) -> Tuple[bool, str]:
        """
        Execute a single test for a given trajectory sample of a system.
        Args:
            test_fn: the attractor test function to execute
            dyst_name: name of the dyst
            traj_sample: the trajectory sample to test
            sample_idx: index of the sample

        Returns:
            bool: True if the test passed, False otherwise
        """
        original_func = (
            test_fn.func if isinstance(test_fn, functools.partial) else test_fn
        )
        func_name = original_func.__name__
        func_params = list(inspect.signature(original_func).parameters.keys())
        can_plot = all(param in func_params for param in ["plot_save_dir", "plot_name"])
        if can_plot and self.plot_save_dir is not None:
            plot_save_dir = os.path.join(self.plot_save_dir, dyst_name)
            test_fn = functools.partial(
                test_fn,
                plot_save_dir=plot_save_dir,
                plot_name=f"{dyst_name}_sample_{sample_idx}",
            )
        # call test_fn on trajectory excluding transient time (burn-in time)
        transient_time = int(traj_sample.shape[1] * self.transient_time_frac)
        status = test_fn(traj_sample[:, transient_time:])
        return status, func_name

    def _filter_dyst(
        self, dyst_name: str, all_traj: np.ndarray, first_sample_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split all trajectories of a given dyst into valid and failed trajectories
        Args:
            dyst_name: name of the dyst
            all_traj: all trajectories of a given dyst
            first_sample_idx: index of the first sample of the dyst

        Returns:
            valid_attractor_trajs: valid trajectories of dyst_name
            failed_attractor_trajs: failed trajectories of dyst_name
        """
        # for each trajectory sample for a given system dyst_name
        valid_attractor_trajs = []
        failed_attractor_trajs = []
        for i, traj_sample in enumerate(all_traj):
            sample_idx = first_sample_idx + i
            # Execute all tests
            status = True
            for test_fn in self.tests:
                status, test_name = self._execute_test_fn(
                    test_fn,
                    dyst_name,
                    traj_sample,
                    sample_idx=sample_idx,
                )
                if not status:
                    # add to failed tests
                    self.failed_checks[dyst_name].append((sample_idx, test_name))
                    self.failed_samples[dyst_name].append(sample_idx)
                    # break upon first failure
                    break

            # if traj sample failed a test, move on to next trajectory sample for this dyst
            if not status:
                # add failed trajectory sample to failed attractor ensemble
                failed_attractor_trajs.append(traj_sample)
                continue

            # if all tests pass, add to valid attractor ensemble
            valid_attractor_trajs.append(traj_sample)
            self.valid_dyst_counts[dyst_name] += 1
            self.valid_samples[dyst_name].append(sample_idx)

        return np.array(valid_attractor_trajs), np.array(failed_attractor_trajs)

    def filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Execute all tests for all trajectory samples in the ensemble, and split the ensemble into valid and failed ensembles.
        Args:
            ensemble: The trajectory ensemble to filter
            first_sample_idx: The index of the first sample for the generated trajectories of the ensemble

        Returns:
            valid_attractor_ensemble: A new ensemble with only the valid trajectories
            failed_attractor_ensemble: A new ensemble with only the failed trajectories
        """
        valid_attractor_ensemble = {}  # Dict[str, np.ndarray]
        failed_attractor_ensemble = {}  # Dict[str, np.ndarray]
        for dyst_name, all_traj in ensemble.items():
            valid_attractor_trajs, failed_attractor_trajs = self._filter_dyst(
                dyst_name, all_traj, first_sample_idx
            )
            if len(failed_attractor_trajs) > 0:
                failed_attractor_ensemble[dyst_name] = failed_attractor_trajs

            # if no valid attractors found, skip this system
            if len(valid_attractor_trajs) == 0:
                print(f"No valid attractor trajectories found for {dyst_name}")
                continue

            if self.verbose:
                print(
                    f"Found {len(valid_attractor_trajs)} valid attractor trajectories for {dyst_name}"
                )
            valid_attractor_ensemble[dyst_name] = valid_attractor_trajs

        return valid_attractor_ensemble, failed_attractor_ensemble

    def _multiprocessed_filter_dyst(
        self,
        dyst_name: str,
        all_traj: np.ndarray,
        first_sample_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]], List[int]]:
        """
        Multiprocessed version of self._filter_dyst without any verbose output
        """
        # book keeping
        failed_checks_samples = []
        valid_samples = []

        valid_attractor_trajs = []
        failed_attractor_trajs = []
        for i, traj_sample in enumerate(all_traj):
            sample_idx = first_sample_idx + i
            # Execute all tests
            status = True
            for test_fn in self.tests:
                status, test_name = self._execute_test_fn(
                    test_fn,
                    dyst_name,
                    traj_sample,
                    sample_idx=sample_idx,
                )
                if not status:
                    failed_check = (sample_idx, test_name)
                    failed_checks_samples.append(failed_check)
                    break
            # if traj sample failed a test, move on to next trajectory sample for this dyst
            if not status:
                failed_attractor_trajs.append(traj_sample)
                continue
            # if all tests pass, add to valid attractor ensemble
            valid_attractor_trajs.append(traj_sample)
            valid_samples.append(sample_idx)
        return (
            np.array(valid_attractor_trajs),
            np.array(failed_attractor_trajs),
            failed_checks_samples,
            valid_samples,
        )

    def multiprocessed_filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Multiprocessed version of self.filter_ensemble
        """
        with Pool() as pool:
            results = pool.starmap(
                self._multiprocessed_filter_dyst,
                [
                    (dyst_name, all_traj, first_sample_idx)
                    for dyst_name, all_traj in ensemble.items()
                ],
            )
        # unpack multiprocessed results
        valid_trajs, failed_trajs, failed_checks, valid_samples = zip(*results)
        # book keeping for failed tests
        for dyst_name, failed_check_lst in zip(list(ensemble.keys()), failed_checks):
            if len(failed_check_lst) > 0:
                self.failed_checks[dyst_name].append(failed_check_lst)
                self.failed_samples[dyst_name].extend(
                    [index for index, _ in failed_check_lst]
                )
        # book keeping for valid samples
        for dyst_name, valid_samples_lst in zip(list(ensemble.keys()), valid_samples):
            if len(valid_samples_lst) > 0:
                self.valid_samples[dyst_name].extend(valid_samples_lst)
                self.valid_dyst_counts[dyst_name] += len(valid_samples_lst)
        # Form the valid and failed ensembles
        valid_ensemble = {
            k: v for k, v in zip(list(ensemble.keys()), valid_trajs) if v.shape[0] > 0
        }
        failed_ensemble = {
            k: v for k, v in zip(list(ensemble.keys()), failed_trajs) if v.shape[0] > 0
        }
        return valid_ensemble, failed_ensemble


def check_boundedness(
    traj: np.ndarray,
    threshold: float = 1e4,
    max_num_stds: float = 10,
    save_plot: bool = False,
    plot_save_dir: Optional[str] = None,
    plot_name: Optional[str] = None,
) -> bool:
    """
    Check if a multi-dimensional trajectory is bounded (not diverging).

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        threshold: Maximum absolute value of the trajectory to consider as diverging.
        max_num_stds: Maximum number of standard deviations from the initial point to consider as diverging.
    Returns:
        bool: False if the system is diverging, True otherwise.
    """
    if np.any(np.abs(traj) > threshold):
        return False

    # Whiten trajectory (i.e. normalize to have zero mean and unit variance)

    whitened_traj = (traj - traj.mean(axis=1)[:, None]) / traj.std(axis=1)[:, None]
    if np.max(np.abs(whitened_traj)) > max_num_stds:
        return False

    # Calculate the Euclidean distance from the first point in the trajectory at each time point
    distance_norms = np.linalg.norm(whitened_traj - whitened_traj[:, 0, None], axis=0)

    if save_plot and plot_save_dir is not None and plot_name is not None:
        os.makedirs(plot_save_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(
            distance_norms,
            bins=100,
            alpha=0.7,
            color="tab:blue",
            edgecolor="black",
        )
        plt.title("Distance Norms of Whitened Trajectory")
        plt.xlabel(r"Distance Norm $||x(t) - x(0)||$")
        plt.ylabel("Counts")
        plt.grid(True)
        plt.savefig(os.path.join(plot_save_dir, f"{plot_name}_distance_norms.png"))
        plt.close()

    is_diverging = np.max(distance_norms) > 2 * max_num_stds

    return not is_diverging


# Function to test if the system goes to a fixed point
def check_not_fixed_point(
    traj: np.ndarray,
    tail_prop: float = 0.05,
    atol: float = 1e-3,
) -> bool:
    """
    Check if the system trajectory converges to a fixed point.
    Actually, this tests the variance decay in the trajectory to detect a fixed point.

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        tail_prop: Proportion of the trajectory to consider for variance comparison.
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    n = traj.shape[1]
    tail = int(tail_prop * n)
    # Compute the Euclidean distance between consecutive points
    distances = np.linalg.norm(np.diff(traj[:, -tail:], axis=1), axis=0)

    if np.allclose(distances, 0, atol=atol):
        return False

    return True


def check_not_trajectory_decay(
    traj: np.ndarray, tail_prop: float = 0.5, atol: float = 1e-3
) -> bool:
    """
    Check if a multi-dimensional trajectory is not decaying.
    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        tail_prop: Proportion of the trajectory to consider for variance comparison.
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    # Check if any dimension of the trajectory is a straight line in the last tail_prop of the trajectory
    n = traj.shape[1]
    tail = int(tail_prop * n)
    for dim in range(traj.shape[0]):
        diffs = np.diff(traj[dim, -tail:])
        if np.allclose(diffs, 0, atol=atol):
            return False
    return True


def check_not_transient(
    traj: np.ndarray, max_transient_prop: float = 0.2, atol: float = 1e-3
) -> bool:
    """
    Check if a multi-dimensional trajectory takes a long time to reach the attractor.
    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        max_transient_prop: first max_transient_prop of the trajectory to consider for checking for transient.
            NOTE: transient is defined loosely here as straight line to hit the attractor.
                We thus search for the presence of straight lines
        atol: Absolute tolerance for detecting a fixed point.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    # Check if any dimension of the trajectory is a straight line in the first max_transient_prop of the trajectory
    n = traj.shape[1]
    transient = int(max_transient_prop * n)
    for dim in range(traj.shape[0]):
        diffs = np.diff(traj[dim, :transient])
        if np.allclose(diffs, 0, atol=atol):
            return False
    return True


# NOTE: this is too brittle to use currently
def check_not_spiral_decay(
    traj: np.ndarray,
    rel_prominence_threshold: Optional[float] = None,
) -> bool:
    """
    Check if a multi-dimensional trajectory is spiraling towards a fixed point.
    Actually, this may also test the variance decay in the trajectory to detect a fixed point.
    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        rel_prominence_threshold: Relative prominence threshold for detecting peaks.
    Returns:
        bool: True if the trajectory does not spiral towards a fixed point, False otherwise.
    """
    n = traj.shape[1]
    # find peaks for each coordinate in the trajectory
    max_peak_indices = [
        find_peaks(t, prominence=rel_prominence_threshold)[0] for t in traj
    ]
    min_peaks_indices = [
        find_peaks(-t, prominence=rel_prominence_threshold)[0] for t in traj
    ]

    # If no peaks are found, just pass this test lazily
    if len(max_peak_indices) == 0 or len(min_peaks_indices) == 0:
        return True

    # Check if peak indices are empty before interpolation
    if any(len(indices) == 0 for indices in max_peak_indices) or any(
        len(indices) == 0 for indices in min_peaks_indices
    ):
        # One or more peak indices are empty. Passing interpolation
        return True

    # Interpolation for envelope
    upper_envelope = np.asarray(
        [np.interp(np.arange(n), i, t[i]) for (i, t) in zip(max_peak_indices, traj)]
    )
    lower_envelope = np.asarray(
        [np.interp(np.arange(n), i, t[i]) for (i, t) in zip(min_peaks_indices, traj)]
    )

    # line fitting, line params shape: [1+1, D]
    line_params = np.polyfit(np.arange(n), traj.T, 1)
    # D x n vector of fitted lines
    line_fit = (
        line_params[0][:, np.newaxis] * np.arange(n) + line_params[1][:, np.newaxis]
    )

    # test if the fitted lines are within the envelope
    within_envelope = (line_fit < upper_envelope) & (line_fit > lower_envelope)
    all_within_envelope = np.all(within_envelope)

    if not all_within_envelope:
        return True

    # test monotonicity of the fitted lines
    diffs = np.diff(line_fit, axis=1)  # D x (n-1)
    monotonic_decrease = np.all(diffs <= 0)

    return not monotonic_decrease


def check_not_limit_cycle(
    traj: np.ndarray,
    tolerance: float = 1e-3,
    min_prop_recurrences: float = 0.0,
    min_counts_per_rtime: int = 100,
    min_block_length: int = 1,
    min_recurrence_time: int = 1,
    enforce_endpoint_recurrence: bool = False,
    save_plot: bool = False,
    plot_save_dir: Optional[str] = None,
    plot_name: Optional[str] = None,
) -> bool:
    """
    Checks if a multidimensional trajectory is collapsing to a limit cycle.

    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        tolerance (float): Tolerance for detecting revisits to the same region in phase space.
        min_prop_recurrences (float): Minimum proportion of the trajectory length that must be recurrences to consider a limit cycle
        min_counts_per_rtime (int): Minimum number of counts per recurrence time to consider a recurrence time as valid
        min_block_length (int): Minimum block length of consecutive recurrence times to consider a recurrence time as valid
        min_recurrence_time (int): Minimum recurrence time to consider a recurrence time as valid
                e.g. Setting min_recurrence_time = 1 means that we can catch when the integration fails (or converges to fixed point)
        enforce_endpoint_recurrence (bool): Whether to enforce that either of the endpoints are recurrences
                e.g. Setting enforce_endpoint_recurrence = True means that we are operating in a stricter regime where we require either
                     the initial or final point to be a recurrence (repeated some time in the trajectory).

    The default args are designed to be lenient, and catch pathological cases beyond purely limit cycles.
        For strict mode, can set e.g. min_prop_recurrences = 0.1, min_block_length=50, min_recurrence_time = 10, enforce_endpoint_recurrence = True,
    Returns:
        bool: True if the trajectory is not collapsing to a limit cycle, False otherwise.
    """
    num_dims, n = traj.shape

    # Step 1: Dimensionality Reduction using PCA (if more than 3 dimensions)
    if num_dims > 3:
        pca = PCA(n_components=3)
        reduced_traj = pca.fit_transform(traj)
    else:
        reduced_traj = traj

    # Step 2: Calculate the pairwise distance matrix, shape should be (N, N)
    dist_matrix = cdist(reduced_traj.T, reduced_traj.T, metric="euclidean").astype(
        np.float16
    )

    # get upper trangular part of matrix, zero out the lower triangular part
    dist_matrix = np.triu(dist_matrix, k=1)

    # Step 3: Get recurrence times from thresholding distance matrix
    recurrence_indices = np.asarray(
        (dist_matrix < tolerance) & (dist_matrix > 0)
    ).nonzero()

    n_recurrences = len(recurrence_indices[0])
    if n_recurrences == 0:
        return True

    if enforce_endpoint_recurrence:
        # check if an eps neighborhood around either n-1 or 0 is in either of the recurrence indices
        eps = 0
        if not any(
            (n - 1) - max(indices) <= eps or min(indices) - 0 <= eps
            for indices in recurrence_indices
        ):
            return True

    # get recurrence times
    recurrence_times = np.abs(recurrence_indices[0] - recurrence_indices[1])
    recurrence_times = recurrence_times[recurrence_times >= min_recurrence_time]

    # Heuristic 1: Check if there are enough recurrences to consider a limit cycle
    n_recurrences = len(recurrence_times)
    if n_recurrences < int(min_prop_recurrences * n):
        # Not enough recurrences to consider a limit cycle.
        return True

    # Heuristic 2: Check if there are enough valid recurrence times
    rtimes_counts = Counter(recurrence_times)
    n_valid_rtimes = sum(
        1 for count in rtimes_counts.values() if count >= min_counts_per_rtime
    )
    if n_valid_rtimes < 1:
        # Not enough valid recurrence times to consider a limit cycle
        return True

    # Heuristic 3: Check if the valid recurrence times are formed of blocks of consecutive timepoints
    if min_block_length > 1:
        rtimes_dict = defaultdict(list)
        block_length = 1
        prev_rtime = None
        prev_t1 = None
        prev_t2 = None
        rtimes_is_valid = False
        num_blocks = 0
        # assuming recurrence_indices[0] is sorted
        for t1, t2 in zip(*recurrence_indices):
            rtime = abs(t2 - t1)
            if rtime < min_recurrence_time:
                continue
            if (
                rtime == prev_rtime
                and abs(t1 - prev_t1) == 1
                and abs(t2 - prev_t2) == 1
            ):
                block_length += 1
            else:
                if block_length > min_block_length:
                    rtimes_dict[prev_rtime].append(block_length)
                    num_blocks += 1
                block_length = 1
            prev_t1, prev_t2, prev_rtime = t1, t2, rtime
            if block_length > min_block_length * 2:
                # enough is enough
                rtimes_is_valid = True
                break
            if num_blocks >= 2:  # if valid, save computation and break
                rtimes_is_valid = True
                break
        if not rtimes_is_valid:
            return True

    # Plot the recurrence times as histogram and 3D trajectory
    # NOTE: at this point, this test has detected a limit cycle (return False)
    if save_plot and plot_save_dir is not None and plot_name is not None:
        os.makedirs(plot_save_dir, exist_ok=True)
        dyst_name = plot_name.split("_")[0]
        plot_name = f"{plot_name}_recurrence_times_FAILED"

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

        ax1.hist(recurrence_times, bins=100, edgecolor="black")
        ax1.set_xlabel("Recurrence Time")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Recurrence Times")
        ax1.grid(True)

        xyz = traj[:3, :]
        xyz1 = xyz[:, : int(n / 2)]
        xyz2 = xyz[:, int(n / 2) :]
        ic_point = traj[:3, 0]
        final_point = traj[:3, -1]
        ax2 = fig.add_subplot(312, projection="3d")
        ax2.plot(*xyz1, alpha=0.5, linewidth=1, color="tab:blue")
        ax2.plot(*xyz2, alpha=0.5, linewidth=1, color="tab:orange")
        ax2.scatter(*ic_point, marker="*", s=100, alpha=0.5, color="tab:blue")
        ax2.scatter(*final_point, marker="x", s=100, alpha=0.5, color="tab:orange")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")  # type: ignore
        ax2.set_title(dyst_name)

        ax3 = fig.add_subplot(313)
        X, Y = np.meshgrid(
            np.arange(dist_matrix.shape[0]), np.arange(dist_matrix.shape[1])
        )
        pcolormesh = ax3.pcolormesh(
            X,
            Y,
            dist_matrix,
            cmap="viridis_r",
            shading="auto",
            norm=colors.LogNorm(),
        )
        plt.colorbar(pcolormesh, ax=ax3)
        ax3.scatter(
            recurrence_indices[0],
            recurrence_indices[1],
            color="black",
            s=20,
            alpha=0.5,
        )
        ax3.set_title("Recurrence Distance Matrix")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Time")
        ax3.set_aspect("equal")

        plot_save_path = os.path.join(plot_save_dir, f"{plot_name}.png")
        plt.savefig(plot_save_path, dpi=300)
        plt.tight_layout()
        plt.close()

    # Step 4: Identify if recurrences are periodic by looking at concentration
    return False


def check_lyapunov_exponent(
    traj: np.ndarray,
) -> bool:
    """
    Check if the Lyapunov exponent of the trajectory is greater than 1.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
    Returns:
        bool: False if the Lyapunov exponent is less than 1, True otherwise.
    """
    lyapunov_exponent = max_lyapunov_exponent_rosenstein(traj.T)
    if lyapunov_exponent < 0:
        # The trajectory does not exhibit chaotic behavior.
        return False
    return True


def check_power_spectrum_1d(
    signal,
    rel_peak_height_threshold: float = 1e-5,
    rel_prominence_threshold: Optional[float] = 1e-5,  # None,
    save_plot: bool = False,
    plot_save_dir: Optional[str] = None,
    plot_name: Optional[str] = None,
) -> bool:
    """
    Analyzes the power spectrum of a 1D signal to find significant peaks and plots the spectrum on a log scale.
    Args:
        signal (array-like): The input 1D signal.
        peak_height_threshold (float): Minimum relative height of a peak to be considered significant.
        prominence_threshold (float): Minimum prominence of a peak to be considered significant.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    n = len(signal)

    # Compute the FFT and the power spectrum
    fft_values = fft(signal)
    power_spectrum = np.abs(fft_values) ** 2  # type: ignore
    freqs = fftfreq(n)

    # Consider only positive frequencies
    pos_freqs = freqs[freqs > 0]
    pos_power = power_spectrum[freqs > 0]

    # Ensure pos_power is numeric
    pos_power = np.asarray(pos_power, dtype=float)

    # Find significant peaks in the power spectrum
    max_peak_height = np.max(pos_power)
    baseline = np.min(pos_power)
    peak_height_threshold = rel_peak_height_threshold * max_peak_height
    peak_height_threshold = max(peak_height_threshold, baseline)
    if rel_prominence_threshold is not None:
        prominence_threshold = rel_prominence_threshold * max_peak_height
        prominence_threshold = max(prominence_threshold, baseline)
    else:
        prominence_threshold = None
    peak_indices, _ = find_peaks(
        pos_power, height=peak_height_threshold, prominence=prominence_threshold
    )
    peak_frequencies = pos_freqs[peak_indices]
    peak_powers = pos_power[peak_indices]

    n_significant_peaks = len(peak_frequencies)

    status = True
    # Heuristic Interpretation of the Power Spectrum
    if n_significant_peaks < 3:
        # The power spectrum suggests a fixed point or a simple periodic attractor (few peaks)
        status = False
    elif n_significant_peaks > 10:
        # The power spectrum suggests a chaotic attractor (many peaks with broad distribution)
        status = True
    else:
        # The system appears to have a quasi-periodic or more complex attractor (intermediate peaks)
        status = True  # this test is intentionally loose

    if status:
        return True

    # Only plot if the test failed
    if save_plot and plot_save_dir is not None and plot_name is not None:
        os.makedirs(plot_save_dir, exist_ok=True)
        plot_name = f"{plot_name}_power_spectrum"
        if not status:
            plot_name = f"{plot_name}_FAILED"
        # Plot the power spectrum on a logarithmic scale
        plt.figure(figsize=(10, 6))
        plt.plot(pos_freqs, pos_power, label="Power Spectrum")
        plt.scatter(
            peak_frequencies,
            peak_powers,
            color="red",
            label="Significant Peaks",
            zorder=5,
        )
        plt.yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title("Power Spectrum with Significant Peaks")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.savefig(os.path.join(plot_save_dir, f"{plot_name}.png"), dpi=300)
        plt.close()
        print(f"Saved plot to {os.path.join(plot_save_dir, f'{plot_name}.png')}")

    return status


def check_power_spectrum(
    traj: np.ndarray,
    rel_peak_height_threshold: float = 1e-5,
    rel_prominence_threshold: Optional[float] = None,
    save_plot: bool = False,
    plot_save_dir: Optional[str] = None,
    plot_name: Optional[str] = None,
) -> bool:
    """
    Check if a multi-dimensional trajectory has a power spectrum that is significant.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
    Returns:
        bool: True if the power spectrum is significant, False otherwise.
    """
    num_dims = traj.shape[0]
    for d in range(num_dims):
        x = traj[d, :]
        status = check_power_spectrum_1d(
            x,
            rel_peak_height_threshold=rel_peak_height_threshold,
            rel_prominence_threshold=rel_prominence_threshold,
            save_plot=save_plot,
            plot_save_dir=plot_save_dir,
            plot_name=plot_name,
        )
        # NOTE: some systems have a periodic driver in the last dimension, which will make this test fail
        # in that case, we can use the Lyapunov exponent test to check for chaos
        if not status:
            return check_lyapunov_exponent(traj)
    return True
