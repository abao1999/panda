"""
Callbacks to check if generated trajectories are valid attractors
"""

import functools
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from dysts.analysis import max_lyapunov_exponent_rosenstein
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import t as student_t
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, kpss

from dystformer.utils import plot_trajs_multivariate, plot_trajs_univariate

BURN_TIME = 200


@dataclass
class EnsembleCallbackHandler:
    """
    Class to handle callbacks for checking if generated trajectories are valid attractors.
    """

    verbose: int = 1
    plot_save_dir: Optional[str] = None

    def __post_init__(self):
        self.callbacks = []  # List[Callable]
        self.failed_checks = defaultdict(list)  # Dict[str, List[Tuple[int, str]]]
        self.valid_dyst_counts = defaultdict(int)  # Dict[str, int]
        if self.plot_save_dir is not None:
            os.makedirs(self.plot_save_dir, exist_ok=True)

    def add_callback(self, callback):
        """
        Add a callback to the list of callbacks.
        """
        assert callable(callback), "Callback must be a callable function"
        self.callbacks.append(callback)

    def plot_phase_space(
        self,
        ensemble: Dict[str, np.ndarray],
        save_dir="tests/figs",
        plot_univariate: bool = False,
    ) -> None:
        """
        Plot the phase space of the attractor for all systems in the ensemble.
        """
        for dyst_name, all_traj in ensemble.items():
            plot_trajs_multivariate(
                all_traj, save_dir=save_dir, plot_name=dyst_name, plot_2d_slice=False
            )
            if plot_univariate:
                num_dims = all_traj.shape[1]
                for dim_idx in range(3 if num_dims > 3 else num_dims):
                    plot_trajs_univariate(
                        all_traj,
                        selected_dim=dim_idx,
                        save_dir=save_dir,
                        plot_name=f"{dyst_name}_univariate_dim{dim_idx}",
                    )

    def _get_callback_name(self, callback: Callable) -> str:
        """
        Get the name of the callback test function
        """
        if isinstance(callback, functools.partial):
            # case when the test is a functools.partial
            callback_name = callback.func.__name__  # Access the wrapped function's name
        else:
            callback_name = callback.__name__  # Directly access the function's name
        if self.verbose >= 1:
            print(f"Executing callback: {callback_name}")
        return callback_name

    def _execute_callback(
        self,
        callback: Callable,
        dyst_name: str,
        traj_sample: np.ndarray,
        sample_idx: int,
    ) -> bool:
        """
        Execute a single callback for a given trajectory sample of a system.
        """
        if self.verbose >= 2:
            callback = functools.partial(callback, verbose=True)
        # Execute the callback test
        status = callback(traj_sample)  # TODO: add dyst_name optional argument
        # book keeping
        if not status:
            callback_name = self._get_callback_name(callback)
            if self.verbose >= 1:
                print(f"FAILED {callback_name} for {dyst_name} at sample {sample_idx}")
            # add to failed checks
            self.failed_checks[dyst_name].append((sample_idx, callback_name))
        return status

    def filter_ensemble(
        self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Execute all callbacks for all trajectory samples in the ensemble. Save the valid trajectories to the valid_attractor_ensemble dictionary.
        """
        valid_attractor_ensemble = {}  # Dict[str, np.ndarray]
        # keep track of the failure too, for debugging purposes
        failed_attractor_ensemble = {}  # Dict[str, np.ndarray]
        # assert first_sample_idx >= 0, "First sample index must be a non-negative integer."
        for dyst_name, all_traj in ensemble.items():
            # for each trajectory sample for a given system dyst_name
            valid_attractor_trajs = []
            failed_attractor_trajs = []
            for i, traj_sample in enumerate(all_traj):
                sample_idx = first_sample_idx + i

                # Make sure traj_sample is a 2D array
                if (
                    traj_sample.ndim == 1
                ):  # handles case where just a single trajectory sample was stored in dict
                    traj_sample = np.expand_dims(traj_sample, axis=0)
                if self.verbose >= 1:
                    print(
                        f"Checking trajectory sample {sample_idx} for {dyst_name}, with shape {traj_sample.shape}"
                    )

                # Execute all callbacks
                for callback in self.callbacks:
                    status = self._execute_callback(
                        callback,
                        dyst_name,
                        traj_sample,
                        sample_idx=sample_idx,
                    )
                    # break upon first failure
                    if not status:
                        break

                # if traj sample failed a check, move on to next trajectory sample for this dyst
                if not status:
                    # add failed trajectory sample to failed attractor ensemble
                    failed_attractor_trajs.append(traj_sample)
                    continue

                # if all checks pass, add to valid attractor ensemble
                valid_attractor_trajs.append(traj_sample)
                self.valid_dyst_counts[dyst_name] += 1

            if len(failed_attractor_trajs) > 0:
                # Add the failed attractor trajectories for this dyst_name system to the failed ensemble
                failed_attractor_ensemble[dyst_name] = np.array(failed_attractor_trajs)

            # if no valid attractors found, skip this system
            if len(valid_attractor_trajs) == 0:
                print(f"No valid attractor trajectories found for {dyst_name}")
                continue

            if self.verbose >= 1:
                print(
                    f"Found {len(valid_attractor_trajs)} valid attractor trajectories for {dyst_name}"
                )
            # Add the valid attractor trajectories for this dyst_name system to the valid ensemble
            valid_attractor_ensemble[dyst_name] = np.array(valid_attractor_trajs)

        return valid_attractor_ensemble, failed_attractor_ensemble


### Start of attractor checks (callbacks) ###
def check_no_nans(traj: np.ndarray, verbose: bool = False) -> bool:
    """
    Check if a multi-dimensional trajectory contains NaN values.
    Returns:
        bool: False if the system contains NaN values, True otherwise.
    """
    if np.isnan(traj).any():
        if verbose:
            print("Trajectory contains NaN values.")
        return False
    return True


def check_boundedness(
    traj: np.ndarray,
    abs_threshold: float = 1e4,
    max_num_stds: float = 1e2,
    verbose: bool = False,
) -> bool:
    """
    Check if a multi-dimensional trajectory is bounded (not diverging).

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        abs_threshold: Maximum absolute value of the trajectory to consider as diverging.
        max_num_stds: Maximum number of standard deviations from the initial point to consider as diverging.
    Returns:
        bool: False if the system is bounded, True otherwise.
    """

    # NOTE: this should have already been caught by integration instability event
    if np.any(np.abs(traj) > abs_threshold):
        if verbose:
            print("Trajectory appears to be diverging.")
        return False

    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    # Initial point (reference point)
    initial_point = traj[:, 0, None]

    # Calculate the Euclidean distance from the first point in the trajectory at each time point
    distances = np.linalg.norm(traj - initial_point, axis=0)

    # Check if the trajectory is diverging
    is_diverging = np.max(distances) > max_num_stds * np.std(distances)

    if verbose:
        print(f"Maximum distance from initial point: {np.max(distances)}")
        print(f"Standard deviation of distances: {np.std(distances)}")
        if is_diverging:
            print("Trajectory appears to be diverging.")
        else:
            print("Trajectory does not appear to be diverging.")

    return not is_diverging


# Function to check if the system goes to a fixed point
def check_not_fixed_point(
    traj: np.ndarray,
    tail_prop: float = 0.05,
    atol: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """
    Check if the system trajectory converges to a fixed point.
    Actually, this checks the variance decay in the trajectory to detect a fixed point.

    Args:
        variables (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        min_variance_threshold (float): Minimum variance threshold for detecting a fixed point.
        window_prop (float): Proportion of the trajectory to consider for variance comparison.
    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    n = traj.shape[1]
    tail = int(tail_prop * n)
    # Check if the trajectory has collapsed to a fixed point
    # if np.allclose(traj[:, -1], traj[:, -2], atol=atol):
    # Compute the Euclidean distance between consecutive points
    distances = np.linalg.norm(np.diff(traj[:, -tail:], axis=1), axis=0)

    # Check if all distances are below the specified tolerance
    # if np.allclose(traj[:, -tail - 1 : -1] - traj[:, -tail:], 0, atol=atol):
    if np.allclose(distances, 0, atol=atol):
        if verbose:
            print(
                f"System may have collapsed to a fixed point, determined with atol={atol}."
            )
        return False
    return True


def check_not_variance_decay(
    traj: np.ndarray,
    tail_prop: float = 0.05,
    min_variance_threshold: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """
    Check if a multi-dimensional trajectory is approaching a fixed point.
    Actually, this checks the variance decay in the trajectory to detect a fixed point.

    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        last_n (int): Number of points to consider for the variance decay check.
        min_variance_threshold (float): Minimum variance threshold for detecting a fixed point.
        variance_num_ooms_threshold (float): Number of order of magnitudes threshold for variance comparison between initial and final segments.
    Returns:
        bool: False if the system has monotonically decaying variance, True otherwise.
    """
    if tail_prop < 0 or tail_prop > 1:
        raise ValueError("tail_prop must be between 0 and 1.")
    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    n = traj.shape[1]
    last_n = int(tail_prop * n)

    # Check if the last_n points have near zero variance
    final_segment = traj[:, -last_n:]  # Last n points
    final_variance = np.var(final_segment, axis=1)
    near_zero_variance = np.any(final_variance < min_variance_threshold)
    if near_zero_variance:
        if verbose:
            print("The system trajectory appears to approach a fixed point.")
        return False

    # # split signal into K bins and compute variance in each bin and fit a line to the variances
    # # if the slope of the line is negative, the variance is decreasing near monotonically
    # # if the slope is positive, the variance is increasing near monotonically
    # # if the slope is zero, the variance is constant near monotonically
    # num_bins = 10
    # bin_size = int(n / num_bins)
    # variances = []
    # for i in range(num_bins):
    #     start = i * bin_size
    #     end = (i + 1) * bin_size
    #     bin = traj[:, start:end]
    #     variances.append(np.var(bin, axis=1))
    # slopes = np.polyfit(np.arange(num_bins), np.array(variances), 1)[0]
    # if slopes < 0:
    #     if verbose:
    #         print("The system trajectory appears to approach a fixed point.")
    #     return False
    return True


def check_not_spiral_decay(
    traj: np.ndarray,
    rel_prominence_threshold: Optional[float] = None,
    verbose: bool = False,
) -> bool:
    """
    Check if a multi-dimensional trajectory is spiraling towards a fixed point.
    Actually, this may also check the variance decay in the trajectory to detect a fixed point.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
    Returns:
        bool: True if the trajectory does not spiral towards a fixed point, False otherwise.
    """
    # Split trajectory into two halves and check variance is not too low in second half compared to first half
    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    n = traj.shape[1]
    # find peaks for each coordinate in the trajectory
    max_peak_indices = [
        find_peaks(t, prominence=rel_prominence_threshold)[0] for t in traj
    ]
    min_peaks_indices = [
        find_peaks(-t, prominence=rel_prominence_threshold)[0] for t in traj
    ]
    if verbose:
        print("Number of peaks: ", len(max_peak_indices))
        print("Number of peaks: ", len(min_peaks_indices))

    # If no peaks are found, just pass this test lazily
    if len(max_peak_indices) == 0 or len(min_peaks_indices) == 0:
        if verbose:
            print("No peaks found. Passing lazily")
        return True

    # Check if peak indices are empty before interpolation
    if any(len(indices) == 0 for indices in max_peak_indices) or any(
        len(indices) == 0 for indices in min_peaks_indices
    ):
        if verbose:
            print("One or more peak indices are empty. Passing interpolation.")
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

    # check if the fitted lines are within the envelope
    within_envelope = (line_fit < upper_envelope) & (line_fit > lower_envelope)
    all_within_envelope = np.all(within_envelope)

    if not all_within_envelope:
        return True

    # check monotonicity of the fitted lines
    diffs = np.diff(line_fit, axis=1)  # D x (n-1)
    monotonic_decrease = np.all(diffs <= 0)

    return not monotonic_decrease


def check_not_limit_cycle(
    traj: np.ndarray,
    tolerance: float = 1e-3,
    min_recurrences: int = 5,
    verbose: bool = False,
) -> bool:
    """
    Checks if a multidimensional trajectory is collapsing to a limit cycle.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        tolerance (float): Tolerance for detecting revisits to the same region in phase space.
        min_recurrences (int): Minimum number of recurrences to consider a limit cycle.
    Returns:
        bool: True if the trajectory is not collapsing to a limit cycle, False otherwise.
    """
    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    num_dims, n = traj.shape

    # Step 1: Dimensionality Reduction using PCA (if more than 3 dimensions)
    if num_dims > 3:
        pca = PCA(n_components=3)
        reduced_traj = pca.fit_transform(traj)
    else:
        reduced_traj = traj

    # Step 2: Recurrence Detection using Distance Matrix
    dist_matrix = np.linalg.norm(
        reduced_traj[:, np.newaxis] - reduced_traj[np.newaxis, :], axis=-1
    )

    # Consider recurrences within a certain tolerance
    recurrence_indices = np.where(dist_matrix < tolerance)
    recurrence_times = np.diff(recurrence_indices[0])

    # Step 3: Identify if recurrences are periodic
    periodic_recurrences = recurrence_times[
        np.abs(recurrence_times - np.median(recurrence_times)) < tolerance
    ]
    if verbose:
        print("Number of periodic recurrences: ", len(periodic_recurrences))
    # Check if recurrences meet the minimum count
    if len(periodic_recurrences) >= min_recurrences:
        if verbose:
            print(
                "The trajectory suggests a limit cycle with stable periodic behavior."
            )
        return False
    return True


def check_lyapunov_exponent(
    traj: np.ndarray,
    verbose: bool = False,
) -> bool:
    """
    Check if the Lyapunov exponent of the trajectory is greater than 1.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
    Returns:
        bool: False if the Lyapunov exponent is less than 1, True otherwise.
    """
    lyapunov_exponent = max_lyapunov_exponent_rosenstein(traj.T)
    if verbose:
        print(f"Max Lyapunov exponent: {lyapunov_exponent}")
    if lyapunov_exponent < 0:
        if verbose:
            print("The trajectory does not exhibit chaotic behavior.")
        return False
    return True


def check_power_spectrum_1d(
    signal,
    rel_peak_height_threshold: float = 1e-5,
    rel_prominence_threshold: Optional[float] = 1e-5,  # None,
    plot_save_dir: Optional[str] = None,
    plot_name: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Analyzes the power spectrum of a 1D signal to find significant peaks and plots the spectrum on a log scale.
    Args:
        signal (array-like): The input 1D signal.
        peak_height_threshold (float): Minimum relative height of a peak to be considered significant.
        prominence_threshold (float): Minimum prominence of a peak to be considered significant.
        plot_save_dir (str): Directory to save the plot.
        plot_name (str): Name of the plot.
        verbose (bool): Whether to print verbose output.
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

    if verbose:
        # Print the significant peaks
        print("Number of significant peaks:", n_significant_peaks)
        print(f"Significant Peaks Frequencies: {peak_frequencies}")
        print(f"Significant Peaks Powers: {peak_powers}")

    # Heuristic Interpretation of the Power Spectrum
    if n_significant_peaks < 3:
        if verbose:
            print(
                "The power spectrum suggests a fixed point or a simple periodic attractor (few peaks)."
            )
        return False
    elif n_significant_peaks > 10:
        if verbose:
            print(
                "The power spectrum suggests a chaotic attractor (many peaks with broad distribution)."
            )
        # return True
    else:
        if verbose:
            print(
                "The system appears to have a quasi-periodic or more complex attractor (intermediate peaks)."
            )

    if plot_save_dir is not None:
        plot_name = plot_name or "power_spectrum"
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

    return True  # make this check loose


def check_power_spectrum(
    traj: np.ndarray,
    rel_peak_height_threshold: float = 1e-5,
    rel_prominence_threshold: Optional[float] = None,
    plot_save_dir: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """
    Check if a multi-dimensional trajectory has a power spectrum that is significant.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        plot_save_dir (str): Directory to save the plot.
        verbose (bool): Whether to print verbose output.
    Returns:
        bool: True if the power spectrum is significant, False otherwise.
    """
    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    num_dims = traj.shape[0]
    for d in range(num_dims):
        x = traj[d, :]
        status = check_power_spectrum_1d(
            x,
            rel_peak_height_threshold=rel_peak_height_threshold,
            rel_prominence_threshold=rel_prominence_threshold,
            plot_name=f"power_spectrum_dim{d}",
            plot_save_dir=plot_save_dir,
            verbose=verbose,
        )
        if not status:
            if verbose:
                print(f"Power spectrum check failed for dimension {d}")
            return check_lyapunov_exponent(traj, verbose=verbose)
    return True


def check_near_recurrences_backup(
    signal: np.ndarray,
    rel_revisit_threshold: float = 0.1,
    revisit_count: int = 3,
    window_prop: float = 0.5,
    verbose: bool = False,
) -> bool:
    """
    Check if a 1D signal revisits near the starting point multiple times.
    Set a revisit threshold as a multiple (rel_revisit_threshold) of the standard deviation of the signal.
    Look at an intermediate window of the signal[int(window_prop * length(signal)):] to check for revisits.
    Set a target value as the mean of the signal.
    If values in the intermediate window are within the revisit threshold of the target value, count them as revisits.
    If the number of revisits is less than revisit_count, the check fails (not enough revisits)

    Parameters:
    - signal (array-like): The input 1D signal.
    - rel_revisit_threshold (float): Threshold relative distance to consider as revisiting near the starting point.
    - revisit_count (int): Minimum number of times the signal should revisit near the starting point.
    - window_prop (float): Proportion of the signal to consider for revisits.
    """
    signal = np.asarray(signal[BURN_TIME:])  # Exclude the burn-in period
    n = len(signal)

    # target = signal[BURN_TIME:] # starting point after burn time
    target = np.mean(signal)  # mean of signal
    revisit_threshold = rel_revisit_threshold * np.std(signal)
    cutoff = int(window_prop * n)
    intermediate_window = signal[cutoff:]
    revisit_indices = np.where(
        np.abs(intermediate_window - target) < revisit_threshold
    )[0]
    if verbose:
        print("Number of revisits: ", len(revisit_indices))
    if len(revisit_indices) < revisit_count:
        return False

    return True


def check_near_recurrences_1d(
    signal: np.ndarray,
    split_prop: float = 0.5,
    tolerance: float = 1e-3,
    num_periods: Optional[int] = None,
    recurrence_ratio_threshold: Optional[float] = None,
    verbose: bool = False,
) -> bool:
    """
    Check the number of times a 1D signal revisits near a baseline (mean or line fit).
    Args:
        signal (ndarray): 1D array of shape (n,), where n is the number of timepoints.
        split_prop (float): Proportion of the signal to consider for revisits.
        tolerance (float): Tolerance for detecting revisits to the same region in phase space.
        num_periods (int): Number of periods to consider for revisits.
        recurrence_ratio_threshold (float): Minimum ratio of recurrence mismatch to consider as recurring.
    Returns:
        bool: True if the signal has enough near-recurrences, False otherwise.
    """
    if split_prop < 0 or split_prop > 1:
        raise ValueError("split_prop must be between 0 and 1")
    if signal.ndim != 1:
        raise ValueError("signal must be a 1D array")
    n = len(signal)

    # baseline = np.mean(signal)  # mean of signal
    # make baseline a line fit to the signal
    x = np.arange(n)
    coef = np.polyfit(x, signal, 1)
    baseline = coef[0] * x + coef[1]

    # split signal into two segments based on split_prop
    n_split = int(split_prop * n)
    first_segment = signal[:n_split]  # First n_split points
    second_segment = signal[n_split:]  # Second n_split points

    # calculate the absolute deviation of segments from the baseline
    abs_deviation_first = np.abs(first_segment - baseline[:n_split])
    abs_deviation_second = np.abs(second_segment - baseline[n_split:])

    recur_indices_first = np.asarray(abs_deviation_first < tolerance).nonzero()[0]
    recur_indices_second = np.asarray(abs_deviation_second < tolerance).nonzero()[0]

    num_recurs_first = len(recur_indices_first)
    num_recurs_second = len(recur_indices_second)

    if num_recurs_first == 0 or num_recurs_second == 0:
        # reject if both segments have no recurrences
        if verbose:
            print("The signal does not have any recurrences in both segments.")
        return False

    # Check the number of recurrences for the two segments
    if num_periods is not None:
        if num_recurs_first < int(num_periods * split_prop):
            return False
        if num_recurs_second < int(num_periods * split_prop):
            return False
    elif recurrence_ratio_threshold is not None:
        if recurrence_ratio_threshold < 1:
            raise ValueError(
                "recurrence_ratio_threshold must be greater than or equal to 1."
            )
        # reject based on ratio of recurrence mismatch
        recurrence_ratio = num_recurs_first / num_recurs_second
        if recurrence_ratio > recurrence_ratio_threshold:
            if verbose:
                print(
                    f"The signal has more than {recurrence_ratio_threshold}x recurrences in the first segment ({num_recurs_first}) than in the second segment ({num_recurs_second})."
                )
            return False
        elif recurrence_ratio < 1 / recurrence_ratio_threshold:
            if verbose:
                print(
                    f"The signal has more than {recurrence_ratio_threshold}x more recurrences in the second segment ({num_recurs_second}) than in the first segment ({num_recurs_first})."
                )
            return False

    return True


def check_near_recurrences(
    traj: np.ndarray,
    split_prop: float = 0.5,
    tolerance: float = 1e-3,
    num_periods: Optional[int] = None,
    recurrence_ratio_threshold: Optional[float] = None,
    verbose: bool = False,
) -> bool:
    """
    Check if every coordinate in a multi-dimensional trajectory revisits near a baseline (mean or line fit) multiple times.
    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        split_prop (float): Proportion of the signal to consider for revisits.
        tolerance (float): Tolerance for detecting revisits to the same region in phase space.
        num_periods (int): Number of periods to consider for revisits.
        recurrence_ratio_threshold (float): Minimum ratio of recurrence mismatch to consider as recurring.
    Returns:
        bool: True if the trajectory has enough near-recurrences, False otherwise.
    """
    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    num_dims = traj.shape[0]

    for d in range(num_dims):
        if verbose:
            print(f"Checking stationarity for dimension {d} by counting recurrences")
        coord = traj[d, :]
        status = check_near_recurrences_1d(
            coord,
            split_prop=split_prop,
            tolerance=tolerance,
            num_periods=num_periods,
            recurrence_ratio_threshold=recurrence_ratio_threshold,
            verbose=verbose,
        )
        if not status:
            return False
    return True


def check_stationarity(
    traj: np.ndarray,
    verbose: bool = False,
    method: str = "statsmodels",
) -> bool:
    """
    ADF checks for presence of a unit root, with null hypothesis that time_series is non-stationary.
    KPSS checks for stationarity around a constant (or deterministic trend), with null hypothesis that time_series is stationary.
    NOTE: may only be sensible for long enough time horizon.

    Args:
        traj (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        method (str): 'statsmodels' to use statsmodels ADF and KPSS tests,
                    'custom' to use custom tests.
                    None to use custom check
    Returns:
        bool: True if the trajectory is stationary, False otherwise.
    """
    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    # assuming first dimension is the state dimension, shape is (dim, T)
    num_dims = traj.shape[0]

    # If not using recurrence test, check for stationarity using stationarity tests
    for d in range(num_dims):
        if verbose:
            print(f"Checking stationarity for dimension {d}")
        coord = traj[d, :]

        if method == "custom":
            # Use custom ADF and KPSS tests
            status_adf = adf_test(coord)
            status_kpss = kpss_test(coord, regression="c")

        elif method == "statsmodels":
            # Use statsmodels ADF and KPSS tests
            result_adf = adfuller(coord, autolag="AIC")
            result_kpss = kpss(coord, regression="c")
            # Interpret p-values for ADF
            status_adf = 1 if result_adf[1] < 0.05 else 0
            status_kpss = 0 if result_kpss[1] < 0.05 else 1

        else:
            raise ValueError(
                "Invalid method. Choose from 'statsmodels' or 'custom' or 'recurrence'."
            )

        # Aggregate conclusion
        if status_adf and status_kpss:
            if verbose:
                print("Strong evidence for stationarity")
        elif not status_adf and not status_kpss:
            if verbose:
                print("Strong evidence for non-stationarity")
            return False
        else:
            if verbose:
                print("Mixed results, inconclusive")
                print("ADF: ", status_adf)
                print("KPSS: ", status_kpss)
    return True


## Stationarity Checks (TODO: need to fix, does not behave as expected)
def kpss_test(timeseries, regression="c", lags=None):
    """
    TODO: need to fix, does not behave as expected
    Perform KPSS test for stationarity.

    Parameters:
    - timeseries: Array-like, the time series data to test.
    - regression: 'c' for constant (level stationarity) or 'ct' for constant and trend (trend stationarity).
    - lags: Number of lags to include in the test (optional). If None, it defaults to int(12*(n/100)**(1/4)).

    Returns:
    - Test statistic, p-value approximation (not calculated), number of lags, and critical values.
    """

    # Remove the trend (mean or linear trend based on regression parameter)
    n = len(timeseries)

    if regression == "c":
        # Center the series by removing the mean
        detrended = timeseries - np.mean(timeseries)
        critical_values = [0.347, 0.463, 0.574, 0.739]
    elif regression == "ct":
        # Remove a linear trend
        x = np.arange(1, n + 1)
        coef = np.polyfit(x, timeseries, 1)
        trend = coef[0] * x + coef[1]
        detrended = timeseries - trend
        critical_values = [0.119, 0.146, 0.176, 0.216]
    else:
        raise ValueError("regression must be 'c' or 'ct'")

    # Calculate the cumulative sum of residuals
    s = np.cumsum(detrended)

    # Calculate the KPSS test statistic
    eta = np.sum(s**2) / (n**2)

    # Calculate the variance of the residuals
    if lags is None:
        lags = int(12 * (n / 100) ** (1 / 4))  # Default lag selection
    s0 = np.var(detrended, ddof=1)

    # Calculate long-run variance using the Bartlett window
    s_hat = s0
    for i in range(1, lags + 1):
        weight = 1 - i / (lags + 1)
        gamma_i = np.sum(detrended[i:] * detrended[:-i]) / n
        s_hat += 2 * weight * gamma_i

    kpss_statistic = eta / s_hat
    pvals = [0.10, 0.05, 0.025, 0.01]

    p_value = np.interp(kpss_statistic, critical_values, pvals)

    warn_msg = """\
    The test statistic is outside of the range of p-values available in the
    look-up table. The actual p-value is {direction} than the p-value returned.
    """
    if p_value == pvals[-1]:
        warnings.warn(
            warn_msg.format(direction="smaller"),
            Warning,
            stacklevel=2,
        )
    elif p_value == pvals[0]:
        warnings.warn(
            warn_msg.format(direction="greater"),
            Warning,
            stacklevel=2,
        )

    is_stationary = p_value >= 0.05
    return is_stationary


def adf_test(y, max_lag=None):
    """
    TODO: need to fix, does not behave as expected
    Perform Augmented Dickey-Fuller test from scratch.

    Parameters:
    - y: The time series data as a 1D numpy array.
    - max_lag: The maximum number of lags to include in the regression.

    Returns:
    - t_stat: The test statistic value.
    - critical_values: Critical values at 1%, 5%, and 10% significance levels.
    - p_value: The p-value of the test statistic.
    """

    # Step 1: Compute the differences and lagged values
    y_diff = np.diff(y)
    y_lag = y[:-1]

    if max_lag is None:
        max_lag = int(12 * (len(y_diff) / 100) ** (1 / 4))

    # Step 2: Create lagged difference matrix for lagged terms
    X = np.column_stack([y_lag[max_lag - i : -i] for i in range(1, max_lag + 1)])

    # Prepare the regression matrix with a constant term
    X = np.column_stack([np.ones(len(X)), y_lag[max_lag:], X])
    y_diff = y_diff[max_lag:]

    # Step 3: Estimate coefficients using Ordinary Least Squares (OLS)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y_diff
    residuals = y_diff - X @ beta

    # Step 4: Compute the test statistic (t-statistic)
    gamma = beta[1]  # Coefficient for y_{t-1}
    se_gamma = np.sqrt(
        np.sum(residuals**2) / (len(y_diff) - X.shape[1]) / np.sum(X[:, 1] ** 2)
    )
    t_stat = gamma / se_gamma

    # Step 5: Critical values and p-value
    p_value = 2 * (1 - student_t.cdf(abs(t_stat), df=len(y_diff) - X.shape[1]))
    # critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}

    # Determine if the series is stationary based on the 5% significance level
    # is_stationary = t_stat < critical_values['5%']
    is_stationary = p_value < 0.05

    return is_stationary  # strong evidence for stationarity
