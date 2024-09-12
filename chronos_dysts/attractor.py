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

from statsmodels.tsa.stattools import adfuller, kpss

from chronos_dysts.utils import plot_trajs_multivariate, plot_trajs_univariate

BURN_TIME = 200

class EnsembleCallbackHandler:
    def __init__(self, verbose: int = 1):
        self.callbacks = []
        self.verbose = verbose
        self.failed_checks = defaultdict(list)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def check_status_all(self):
        if len(self.failed_checks) > 0:
            print(f"FAILED CHECKS: {self.failed_checks}")
            return False
        print("ALL CHECKS PASSED.")
        return True

    def check_status_dyst(self, dyst_name: str):
        if len(self.failed_checks[dyst_name]) > 0:
            print(f"FAILED CHECKS for {dyst_name}: {self.failed_checks[dyst_name]}")
            return False
        print(f"ALL CHECKS PASSED for {dyst_name}.")
        return True

    def plot_phase_space(
            self, 
            ensemble: Dict[str, np.ndarray], 
            save_dir="tests/figs",
            plot_univariate: bool = False,
    ) -> None:
        for dyst_name, all_traj in ensemble.items():
            plot_trajs_multivariate(all_traj, save_dir=save_dir, plot_name=dyst_name, plot_2d_slice=False)
            if plot_univariate:
                num_dims = all_traj.shape[1]
                for dim_idx in range(3 if num_dims > 3 else num_dims):
                    plot_trajs_univariate(
                        all_traj,
                        selected_dim = dim_idx,
                        save_dir = save_dir,
                        plot_name = f"{dyst_name}_univariate_dim{dim_idx}"
                    )


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
                        break


def check_no_nans(traj: np.ndarray, verbose: bool = False) -> bool:
    """
    Check if a multi-dimensional trajectory contains NaN values.
    """
    if np.isnan(traj).any():
        if verbose: print("Trajectory contains NaN values.")
        return False
    return True

def check_boundedness(
        traj: np.ndarray,
        max_num_stds: float = 1e2,
        verbose: bool = False) -> bool:
    """
    Check if a multi-dimensional trajectory is bounded (not diverging).

    Args:
        traj: np.ndarray of shape (num_dims, num_timepoints), the trajectory data.
        max_num_stds: Maximum number of standard deviations from the initial point to consider as diverging.
    """

    if np.any(np.abs(traj) > 1e3):
        if verbose: print("Trajectory appears to be diverging.")
        return False

    traj = traj[:, BURN_TIME:]  # Exclude the burn-in period
    # Initial point (reference point)
    initial_point = traj[:, 0, None]

    # Calculate the Euclidean distance from the first point in the trajectory at each time point
    distances = np.linalg.norm(traj - initial_point, axis=0)
    if verbose: print("std of distances: ", np.std(distances))
    
    # Check if the trajectory is diverging
    is_diverging = (np.max(distances) > max_num_stds * np.std(distances))
    
    # # NOTE: looking at trends seems brittle, need very long horizon to be reliable
    # if is_diverging: # double check by looking at trend
    #     # Fit a polynomial to the distances to check the trend
    #     time_points = np.linspace(0, 1, distances.size)  # Normalize time to [0, 1]
    #     poly_degree = 1  # Linear fit
    #     poly_coeffs = np.polyfit(time_points, distances, deg=poly_degree)

    #     # Determine the trend by examining the leading coefficient of the polynomial fit
    #     # For linear (degree 1), this would be the slope. For higher degrees, examine leading term.
    #     leading_coefficient = poly_coeffs[0]  # Coefficient of the highest degree term

    #     print("Slope: ", leading_coefficient)

    #     if np.abs(leading_coefficient) < np.std(distances): # TODO: does this make sense?
    #         is_diverging = False  # Reverse the divergence status

    if verbose:
        print(f"Maximum distance from initial point: {np.max(distances)}")
        if is_diverging:
            print("Trajectory appears to be diverging.")
        else:
            print("Trajectory does not appear to be diverging.")

    return not is_diverging


# Function to check if the system goes to a fixed point
def check_not_fixed_point(
        traj: np.ndarray,
        min_variance_threshold: float = 1e-3,
        window_prop: float = 0.5,
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
    # Check if the trajectory has collapsed to a fixed point 
    if np.allclose(traj[:, -1], traj[:, -2], atol=1e-3):
    # if np.allclose(traj[:, -99:-50], traj[:, -50:-1], atol=1e-3):
        if verbose: print("System may have collapsed to a fixed point.")
        return False

    # Check of the last 100 points have near zero variance
    final_segment = traj[:, -100:]  # Last 100 points
    variance = np.var(final_segment, axis=1)
    near_zero_variance = np.any(variance < min_variance_threshold) # maybe use all?
    if verbose: print(f"Variance of state variables over the final segment: {variance}")
    if near_zero_variance:
        if verbose: print("The system trajectory appears to approach a fixed point.")
        return False

    # Split trajectory into two halves and check variance is not too low in second half compared to first half
    intermediate_window = traj[:, BURN_TIME:]  # Exclude the burn-in period
    n = intermediate_window.shape[1]
    cutoff = int(window_prop * n)
    rolling_variance_first = np.var(intermediate_window[:, :cutoff], axis=1)
    rolling_variance_second = np.var(intermediate_window[:, cutoff:], axis=1)
    if verbose:
        print("rolling_variance_first: ", rolling_variance_first)
        print("rolling_variance_second: ", rolling_variance_second)
    if np.all(rolling_variance_second < 0.2 * rolling_variance_first):
        if verbose: print("Variance is decaying in the second half of the trajectory.")
        return False

    return True

from sklearn.decomposition import PCA
def check_not_limit_cycle(trajectory, tolerance=1e-3, min_recurrences=5, verbose=False):
    """
    Checks if a multidimensional trajectory is collapsing to a limit cycle.

    Parameters:
    - trajectory (array-like): 2D array where rows are time steps and columns are dimensions.
    - tolerance (float): Tolerance for detecting revisits to the same region in phase space.
    - min_recurrences (int): Minimum number of recurrences to consider a limit cycle.

    Returns:
    - bool: False if the trajectory is collapsing to a limit cycle, otherwise True.
    """
    trajectory = np.asarray(trajectory)
    num_dims, n = trajectory.shape

    # Step 1: Dimensionality Reduction using PCA (if more than 3 dimensions)
    if num_dims > 3:
        pca = PCA(n_components=3)
        reduced_trajectory = pca.fit_transform(trajectory)
    else:
        reduced_trajectory = trajectory

    # Step 2: Recurrence Detection using Distance Matrix
    dist_matrix = np.linalg.norm(reduced_trajectory[:, np.newaxis] - reduced_trajectory[np.newaxis, :], axis=-1)
    
    # Consider recurrences within a certain tolerance
    recurrence_indices = np.where(dist_matrix < tolerance)
    recurrence_times = np.diff(recurrence_indices[0])

    # Step 3: Identify if recurrences are periodic
    periodic_recurrences = recurrence_times[np.abs(recurrence_times - np.median(recurrence_times)) < tolerance]
    if verbose: print("Number of periodic recurrences: ", len(periodic_recurrences))
    # Check if recurrences meet the minimum count
    if len(periodic_recurrences) >= min_recurrences:
        if verbose:
            print("The trajectory suggests a limit cycle with stable periodic behavior.")
        return False
    return True


from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
def check_power_spectrum_1d(
    signal, 
    rel_peak_height_threshold: float = 1e-5, 
    rel_prominence_threshold: Optional[float] = 1e-5, # None,
    plot_name: str = "power_spectrum",
    plot_save_dir: str = "tests/figs",
    verbose: bool = False
) -> bool:
    """
    Analyzes the power spectrum of a 1D signal to find significant peaks and plots the spectrum on a log scale.

    Parameters:
    - signal (array-like): The input 1D signal.
    - peak_height_threshold (float): Minimum relative height of a peak to be considered significant.
    - prominence_threshold (float): Minimum prominence of a peak to be considered significant.
    """
    # Convert the signal to a numpy array
    signal = np.asarray(signal[BURN_TIME:])  # Exclude the burn-in period
    n = len(signal)
    
    # Compute the FFT and the power spectrum
    fft_values = fft(signal)
    power_spectrum = np.abs(fft_values)**2
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
                        pos_power, 
                        height=peak_height_threshold, 
                        prominence=prominence_threshold
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
        if verbose: print("The power spectrum suggests a fixed point or a simple periodic attractor (few peaks).")
        return False
    elif n_significant_peaks > 10:
        if verbose: print("The power spectrum suggests a chaotic attractor (many peaks with broad distribution).")
        # return True
    else:
        if verbose: print("The system appears to have a quasi-periodic or more complex attractor (intermediate peaks).")
    
    if plot_save_dir is not None:
        # Plot the power spectrum on a logarithmic scale
        plt.figure(figsize=(10, 6))
        plt.plot(pos_freqs, pos_power, label='Power Spectrum')
        plt.scatter(peak_frequencies, peak_powers, color='red', label='Significant Peaks', zorder=5)
        plt.yscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title('Power Spectrum with Significant Peaks')
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.savefig(os.path.join(plot_save_dir, f"{plot_name}.png"), dpi=300)

    return True # make this check loose

def check_power_spectrum(
        traj: np.ndarray,
        plot_save_dir: Optional[str] = None,
        verbose: bool = False,
) -> bool:
    num_dims = traj.shape[0]
    for d in range(num_dims):
        x = traj[d, :]
        status = check_power_spectrum_1d(
            x, 
            plot_name=f"power_spectrum_dim{d}",
            plot_save_dir=plot_save_dir, 
            verbose=verbose,
        )
        if not status:
            if verbose: print(f"Power spectrum check failed for dimension {d}")
            return False
    return True

def check_stationarity(
        traj: np.ndarray,
        verbose: bool = False,
        method: str = 'recurrence',
        ) -> bool:
    """
    ADF checks for presence of a unit root, with null hypothesis that time_series is non-stationary.
    KPSS checks for stationarity around a constant (or deterministic trend), with null hypothesis that time_series is stationary.
    NOTE: may only be sensible for long enough time horizon.
    For this reason, we default to a 'recurrence' test, which checks if the trajectory revisits near a target point multiple times.

    Args:
        traj (ndarray): 2D array of shape (dim, T), where each row is a time series.
        method (str): 'statsmodels' to use statsmodels ADF and KPSS tests,
                    'custom' to use custom tests.
                    None to use custom check
    """
    num_dims = traj.shape[0] # assuming first dimension is the state dimension, shape is (dim, T)

    if method == 'recurrence':
        for d in range(num_dims):
            if verbose: print(f"Checking stationarity for dimension {d}")
            coord = traj[d, :]
            status = check_recurrence(
                coord, 
                rel_revisit_threshold=0.2, 
                revisit_count=3,
                window_prop=0.5,
                verbose=verbose,
            )
            print(f"status for dim {d}: {status}")
            if not status:
                return False
        return True

    for d in range(num_dims):
        if verbose: print(f"Checking stationarity for dimension {d}")
        coord = traj[d, :]

        if method == 'custom':
            # Use custom ADF and KPSS tests
            status_adf = adf_test(coord)
            status_kpss = kpss_test(coord, regression='c')

        elif method == 'statsmodels':
            # Use statsmodels ADF and KPSS tests
            result_adf = adfuller(coord, autolag="AIC")
            result_kpss = kpss(coord, regression='c')
            # Interpret p-values for ADF
            status_adf = 1 if result_adf[1] < 0.05 else 0
            status_kpss = 0 if result_kpss[1] < 0.05 else 1
        
        else:
            raise ValueError("Invalid method. Choose from 'statsmodels' or 'custom' or 'recurrence'.")

        # Aggregate conclusion
        if status_adf and status_kpss:
            if verbose: print("Strong evidence for stationarity")
        elif not status_adf and not status_kpss:
            if verbose: print("Strong evidence for non-stationarity")
            return False
        else:
            if verbose:
                print("Mixed results, inconclusive")
                print("ADF: ", status_adf)
                print("KPSS: ", status_kpss)
    return True


# from scipy.stats import linregress
# from scipy.signal import detrend
def check_recurrence(
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
    
    # # Criterion 1: Trend check # NOTE: only reliable for long time horizon
    # detrended_signal = detrend(signal)
    # slope, _, _, p_value, _ = linregress(np.arange(n), signal)
    # print("Slope: ", slope)
    # if p_value < 0.05:  # Significance level for trend
    #     print("Trend check failed")
    #     return False

    # Criterion 2: Recurrence check
    # target = signal[BURN_TIME:] # starting point after burn time
    target = np.mean(signal) # mean of signal
    revisit_threshold = rel_revisit_threshold * np.std(signal)
    cutoff = int(window_prop * n)
    intermediate_window = signal[cutoff:]
    revisit_indices = np.where(np.abs(intermediate_window - target) < revisit_threshold)[0]
    if verbose: print("Number of revisits: ", len(revisit_indices))
    if len(revisit_indices) < revisit_count:
        return False

    return True



## Stationarity Checks (TODO: need to fix, does not behave as expected)
def kpss_test(timeseries, regression='c', lags=None):
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