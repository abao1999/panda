from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.arima.model import ARIMA

from dystformer.utils import safe_standardize


@dataclass
class MeanBaseline:
    prediction_length: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forecast the mean of the context

        Args:
            x: (batch_size, num_timesteps, num_features) numpy array

        Returns:
            (batch_size, prediction_length, num_features) numpy array
        """
        return np.mean(x, axis=1, keepdims=True) * np.ones(
            (x.shape[0], self.prediction_length, x.shape[2])
        )


@dataclass
class FourierBaseline:
    prediction_length: int

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forecast the Fourier series of the context

        Args:
            x: (batch_size, num_timesteps, num_features) numpy array

        Returns:
            (batch_size, prediction_length, num_features) numpy array
        """
        batch_size, context_length, num_features = x.shape
        rfft_vals = np.fft.rfft(safe_standardize(x, axis=1), axis=1)
        ntotal = context_length + self.prediction_length
        reconstructed = np.fft.irfft(rfft_vals, n=ntotal, axis=1)
        return safe_standardize(
            reconstructed[:, context_length:], context=x, axis=1, denormalize=True
        )


@dataclass
class FourierARIMABaseline:
    prediction_length: int
    order: tuple[int, int, int] = (4, 1, 4)
    num_fourier_terms: int = 8
    min_period: int = 20
    max_period: int = 200

    def _estimate_period(self, signal: np.ndarray) -> np.ndarray:
        """Estimate the dominant period from the signal using autocorrelation.

        Args:
            signal: (batch_size, n_points,) numpy array

        Returns:
            Array of shape (batch_size,) containing estimated periods
        """
        batch_size = signal.shape[0]
        periods = np.zeros(batch_size)

        for i in range(batch_size):
            # Normalize signal
            x = signal[i] - np.mean(signal[i])
            x = x / (np.std(x) + 1e-8)

            # Compute autocorrelation
            corr = np.correlate(x, x, mode="full")
            corr = corr[len(x) - 1 :]

            # Find peaks in autocorrelation

            peaks, _ = find_peaks(corr, distance=self.min_period)

            if len(peaks) > 0:
                # Use first strong peak after min_period
                valid_peaks = peaks[peaks <= self.max_period]
                if len(valid_peaks) > 0:
                    periods[i] = valid_peaks[0]
                else:
                    periods[i] = self.max_period
            else:
                periods[i] = self.max_period

        return periods

    def _create_fourier_features(
        self,
        signal: np.ndarray,
        n_points: int,
        period: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Create Fourier features for seasonal decomposition.

        Args:
            signal: (batch_size, n_points,) numpy array
            n_points: Number of time points
            period: Fundamental period for Fourier terms. If None, estimates from data

        Returns:
            Array of shape (batch_size, n_points, 2 * num_fourier_terms) containing sin/cos features
        """
        if period is None:
            period = self._estimate_period(signal)
        elif isinstance(period, (int, float)):
            period = np.ones(signal.shape[0]) * period

        t = np.arange(n_points)
        fourier_features = np.zeros(
            (signal.shape[0], n_points, 2 * self.num_fourier_terms)
        )

        # Add harmonics with phase alignment
        for i in range(self.num_fourier_terms):
            freq = 2 * np.pi * (i + 1) / period
            fourier_features[:, :, 2 * i] = np.sin(freq[..., None] * t)
            fourier_features[:, :, 2 * i + 1] = np.cos(freq[..., None] * t)

        return fourier_features

    def _deseasonalize(
        self, x: np.ndarray, fourier_features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove seasonal components using Fourier features.

        Args:
            x: (batch_size, num_timesteps,) time series
            fourier_features: (batch_size, num_timesteps, 2 * num_fourier_terms) Fourier features

        Returns:
            Deseasonalized time series and coefficients
        """
        # Fit seasonal components
        coeffs = np.linalg.lstsq(fourier_features, x, rcond=None)[0]
        seasonal = fourier_features @ coeffs
        return x - seasonal, coeffs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forecast using Fourier-ARIMA decomposition with pattern continuation.

        Args:
            x: (batch_size, num_timesteps, num_features) numpy array

        Returns:
            (batch_size, prediction_length, num_features) forecasts
        """
        batch_size, context_length, num_features = x.shape
        forecasts = np.zeros((batch_size, self.prediction_length, num_features))

        for i in range(num_features):
            channel = x[:, :, i]

            # Estimate period and create extended features
            period = self._estimate_period(channel)
            total_length = context_length + self.prediction_length

            # Create features for entire sequence
            full_features = self._create_fourier_features(
                channel, total_length, period=period
            )

            context_features = full_features[:, :context_length]
            forecast_features = full_features[:, context_length:]

            for j in range(batch_size):
                try:
                    # Normalize signal
                    mean_val = np.mean(channel[j])
                    std_val = np.std(channel[j])
                    normalized = (channel[j] - mean_val) / (std_val + 1e-8)

                    # Decompose seasonal component
                    deseasonalized, coeffs = self._deseasonalize(
                        normalized, context_features[j]
                    )

                    # Fit ARIMA on deseasonalized data
                    model = ARIMA(deseasonalized, order=self.order)
                    results = model.fit()

                    # Generate ARIMA forecast
                    arima_forecast = results.forecast(self.prediction_length)

                    # Add back seasonal component and denormalize
                    seasonal_forecast = forecast_features[j] @ coeffs
                    combined_forecast = arima_forecast + seasonal_forecast
                    forecasts[j, :, i] = (combined_forecast * std_val) + mean_val

                except Exception:
                    # Fallback to pattern continuation
                    last_period = int(period[j])
                    if last_period > 0:
                        pattern = channel[j, -last_period:]
                        repeats = self.prediction_length // len(pattern) + 1
                        extended = np.tile(pattern, repeats)[: self.prediction_length]
                        forecasts[j, :, i] = extended
                    else:
                        mean_val = np.mean(channel[j])
                        forecasts[j, :, i] = np.ones(self.prediction_length) * mean_val

        return forecasts
