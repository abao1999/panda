"""
Callbacks to check if generated trajectories are valid attractors
"""

import os
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt
import functools
from collections import defaultdict
import warnings

from chronos_dysts.utils import plot_trajs_multivariate


class EnsembleCallbackHandler:
    def __init__(self, verbose: int = 1):
        self.callbacks = []
        self.verbose = verbose
        self.failed_checks = defaultdict(list)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def check_status_all(self):
        if len(self.failed_checks) > 0:
            print(f"Failed checks: {self.failed_checks}")
            return False
        print("All checks passed.")
        return True
            
    def check_status_dyst(self, dyst_name: str):
        if len(self.failed_checks[dyst_name]) > 0:
            print(f"Failed checks for {dyst_name}: {self.failed_checks[dyst_name]}")
            return False
        print(f"All checks passed for {dyst_name}.")
        return True
    
    def plot_phase_space(self, ensemble: Dict[str, np.ndarray], save_dir="tests/figs"):
        for dyst_name, all_traj in ensemble.items():
            for i, traj_sample in enumerate(all_traj):
                if traj_sample.ndim == 2:
                    traj_sample = np.expand_dims(traj_sample, axis=0)
                print(traj_sample.shape)
                plot_trajs_multivariate(traj_sample, save_dir=save_dir, plot_name=dyst_name)


    def execute_callbacks(self, ensemble: Dict[str, np.ndarray], first_sample_idx: int = 0):
        # assert first_sample_idx >= 0, "First sample index must be a non-negative integer."
        for dyst_name, all_traj in ensemble.items():
            for i, traj_sample in enumerate(all_traj):
                sample_idx = first_sample_idx + i
                if traj_sample.ndim == 1: # handles case where just a single trajectory sample was stored in dict
                    traj_sample = np.expand_dims(traj_sample, axis=0)
                if self.verbose >= 1:
                    print(f"Checking trajectory sample {sample_idx} for {dyst_name}, with shape {traj_sample.shape}")
                for callback in self.callbacks:
                    if self.verbose >= 2:
                        callback = functools.partial(callback, verbose=True)
                    # Check if the object is a functools.partial
                    if isinstance(callback, functools.partial):
                        callback_name = callback.func.__name__  # Access the wrapped function's name
                    elif callable(callback):
                        callback_name = callback.__name__  # Directly access the function's name
                    else:
                        raise ValueError("Invalid callback type. Must be a function or functools.partial.")
    
                    if self.verbose >= 1: 
                        print(f"Executing callback: {callback_name}")
                    status = callback(traj_sample)
                    if status == False:
                        self.failed_checks[dyst_name].append((sample_idx, callback_name))
                        # self.failed_checks.append((dyst_name, i, callback_name))
                        # break

def check_no_nans(traj: np.ndarray, verbose: bool = False) -> bool:
    if np.isnan(traj).any():
        if verbose: print("Trajectory contains NaN values.")
        return False
    return True

def check_boundedness(traj: np.ndarray, verbose: bool = False) -> bool:
    # Check for divergence or collapse to a fixed point
    max_value = np.max(np.abs(traj))
    if max_value > 1e3:  # Adjust threshold based on expected ranges
        if verbose:
            print("Warning: System appears to be diverging.")
            print(f"Max value of system variables: {max_value}")
        return False
    return True


# Function to check if the system goes to a fixed point
def check_not_fixed_point(
        traj: np.ndarray, 
        threshold: float = 1e-2,
        verbose: bool = False,
) -> bool:
    """
    Check if the system trajectory converges to a fixed point.

    Args:
        variables (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        threshold (float): Threshold for checking if the system is near a fixed point.

    Returns:
        bool: False if the system is approaching a fixed point, True otherwise.
    """
    # check closeness of last two points in trajectory
    if np.allclose(traj[:, -1], traj[:, -2], atol=1e-3):
        if verbose: print("System may have collapsed to a fixed point.")
        return False
    
    # Check variance of the last segment of the trajectory
    final_segment = traj[:, -100:]  # Last 100 points
    variance = np.var(final_segment, axis=1)
    
    # Additional check using variance to confirm near fixed point behavior
    near_zero_variance = np.all(variance < threshold)
    if verbose:
        print(f"Variance of state variables over the final segment: {variance}")
        if near_zero_variance:
            print("The system trajectory appears to approach a fixed point.")

    return not near_zero_variance


# Function to check if the trajectory goes to a limit cycle
def check_not_limit_cycle(
        traj: np.ndarray, 
        threshold: float = 1e-3,
        plot_save_dir: Optional[str] = None,
        verbose: bool = False,
) -> bool:
    """
    Check if the system trajectory converges to a limit cycle.

    Args:
        variables (ndarray): 2D array of shape (num_vars, num_timepoints), where each row is a time series.
        threshold (float): Threshold for detecting periodicity in the trajectory.

    Returns:
        bool: False if the system appears to have a limit cycle, True otherwise.
    """
    # num_points = traj.shape[1]

    # Calculate distances between successive points in phase space
    diffs = np.diff(traj, axis=1)
    distances = np.linalg.norm(diffs, axis=0)
    
    # Smooth the distances to reduce noise
    smoothed_distances = np.convolve(distances, np.ones(10)/10, mode='valid')
    
    if plot_save_dir is not None:
        # Plot the smoothed distances
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_distances)
        plt.title('Smoothed Distances Between Successive Points in Phase Space')
        plt.xlabel('Time Step')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig(os.path.join(plot_save_dir, "smoothed_distance.png"), dpi=300)

    # Check for periodicity in smoothed distances
    mean_distance = np.mean(smoothed_distances)
    periodic_behavior = np.all(np.abs(smoothed_distances - mean_distance) < threshold)

    if verbose:
        if periodic_behavior:
            print("The trajectory suggests a limit cycle with stable periodic behavior.")
        else:
            print("No clear limit cycle detected; trajectory does not stabilize periodically.")
            
    return not periodic_behavior


def check_power_spectrum(
        traj: np.ndarray, 
        timestep: float = 1.0, 
        plot_save_dir: Optional[str] = None,
        verbose: bool = False,
) -> bool:
    x = traj[0, :]  # Select the first variable for analysis
    # Power Spectrum Analysis
    x_detrended = x - np.mean(x)  # Detrend the data by removing the mean
    power_spectrum = np.abs(np.fft.fft(x_detrended))**2
    frequencies = np.fft.fftfreq(len(x_detrended), d=timestep)

    # since we assume signal is real, we only need to consider positive frequencies
    power_spectrum = power_spectrum[frequencies > 0] # [:len(frequencies)//2]
    frequencies = frequencies[frequencies > 0]

    if plot_save_dir is not None:
        # Plot the power spectrum (positive frequencies only)
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, power_spectrum)
        plt.title('Power Spectrum of X Variable')
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.xlim(0, np.max(frequencies)/2)
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(plot_save_dir, "power_spectrum.png"), dpi=300)


    # Define a threshold for significant peaks
    # TODO: this doesnt find peaks in the power spectrum. Maybe try something like https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    peak_power_threshold = 1e-3 * np.max(power_spectrum)
    significant_peaks = power_spectrum > peak_power_threshold

    if verbose:
        print("Number of significant peaks:", np.sum(significant_peaks))
        print("Significant peak frequencies:", frequencies[significant_peaks])
        # Identify the top peaks in the power spectrum
        peak_indices = np.argsort(power_spectrum, axis=-1)[-5:]  # Get indices of the top 5 peaks
        peak_freqs = frequencies[peak_indices]
        peak_powers = power_spectrum[peak_indices]
        # Display top peak information
        print("Top 5 Peak Frequencies and their Powers:")
        for f, p in zip(peak_freqs, peak_powers):
            print(f"Frequency: {f:.6f}, Power: {p:.4e}")

    # Heuristic Interpretation of the Power Spectrum
    if np.sum(significant_peaks) < 3:
        if verbose: print("The power spectrum suggests a fixed point or a simple periodic attractor (few peaks).")
        return False
    elif np.sum(significant_peaks) > 10:
        if verbose: print("The power spectrum suggests a chaotic attractor (many peaks with broad distribution).")
        return True
    else:
        if verbose: print("The system appears to have a quasi-periodic or more complex attractor (intermediate peaks).")
    return True # make this check loose

## Stationarity Checks
import numpy as np

def kpss_test(timeseries, regression='c', lags=None):
    """
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

    if regression == 'c':
        # Center the series by removing the mean
        detrended = timeseries - np.mean(timeseries)
        critical_values = [0.347, 0.463, 0.574, 0.739]
    elif regression == 'ct':
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



from scipy.stats import t

def adf_test(y, max_lag=None):
    """
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
    X = np.column_stack([y_lag[max_lag - i: -i] for i in range(1, max_lag + 1)])
    
    # Prepare the regression matrix with a constant term
    X = np.column_stack([np.ones(len(X)), y_lag[max_lag:], X])
    y_diff = y_diff[max_lag:]
    
    # Step 3: Estimate coefficients using Ordinary Least Squares (OLS)
    beta = np.linalg.inv(X.T @ X) @ X.T @ y_diff
    residuals = y_diff - X @ beta
    
    # Step 4: Compute the test statistic (t-statistic)
    gamma = beta[1]  # Coefficient for y_{t-1}
    se_gamma = np.sqrt(np.sum(residuals ** 2) / (len(y_diff) - X.shape[1]) / np.sum(X[:, 1] ** 2))
    t_stat = gamma / se_gamma

    # Step 5: Critical values and p-value
    p_value = 2 * (1 - t.cdf(abs(t_stat), df=len(y_diff) - X.shape[1]))
    critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}

    # Determine if the series is stationary based on the 5% significance level
    # is_stationary = t_stat < critical_values['5%']
    is_stationary = p_value < 0.05

    return is_stationary # strong evidence for stationarity


def check_stationarity(traj: np.ndarray, verbose: bool = False) -> bool:
    dim = traj.shape[0] # assuming first dimension is the state dimension, shape is (dim, T)
    for d in range(dim):
        if verbose: print(f"Checking stationarity for dimension {d}")
        coord = traj[d, :]
        status_adf = adf_test(coord)
        status_kpss = kpss_test(coord, regression='c')
        # Aggregate conclusion
        if status_adf and status_kpss:
            if verbose: print("Strong evidence for stationarity")
        elif not status_adf and not status_kpss:
            if verbose: print("Strong evidence for non-stationarity")
            return False
        else:
            if verbose: print("Mixed results, inconclusive")
    return True