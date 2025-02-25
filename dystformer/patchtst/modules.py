"""
Some modules for PatchTST
"""

import torch
import torch.nn as nn
from transformers import PatchTSTConfig


class PatchTSTKernelEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_poly_feats = config.num_poly_feats
        self.poly_feat_degree = config.poly_feat_degree
        self.patch_indices = torch.randint(
            high=config.patch_length,
            size=(self.num_poly_feats, self.poly_feat_degree),
            requires_grad=False,
        )
        self.freq_weights = nn.Parameter(
            torch.randn(config.patch_length, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.freq_biases = nn.Parameter(
            torch.randn(1, 1, 1, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        poly_feats = x[..., self.patch_indices].prod(dim=-1)
        weighted_x = x @ self.freq_weights + self.freq_biases
        rff_feats = torch.cat([torch.sin(weighted_x), torch.cos(weighted_x)], dim=-1)
        return torch.cat([poly_feats, rff_feats], dim=-1)


class PatchTSTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Stolen from Llama
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PatchTSTPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    NOTE: Exposed from original source code. Allow for variable sequence length

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        num_patches = (sequence_length - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (num_patches - 1)
        sequence_start = sequence_length - new_sequence_length

        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, sequence_start:, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(
            dimension=-2, size=self.patch_length, step=self.patch_stride
        )
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


def apply_p_rope_to_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    max_wavelength: int,
    rope_percent: float,
):
    """
    Apply p-rotary positional embeddings to the query and key tensors

    from: https://arxiv.org/pdf/2410.06205
    """
    rope_angles = int(rope_percent * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    fraction = (
        2.0
        * torch.arange(
            0, rope_angles, device=query_states.device, dtype=query_states.dtype
        )
        / head_dim
    )
    timescale = torch.nn.functional.pad(
        max_wavelength**fraction,
        (0, nope_angles),
        mode="constant",
        value=torch.inf,
    )

    # sin, cos: shape (..., 1, seq_len, head_dim//2)
    sinusoid_inp = position_ids[..., None, :, None] / timescale[None, None, :]
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    query_first_half, query_second_half = torch.split(
        query_states, query_states.shape[-1] // 2, dim=-1
    )
    key_first_half, key_second_half = torch.split(
        key_states, key_states.shape[-1] // 2, dim=-1
    )

    query_first_part = query_first_half * cos - query_second_half * sin
    query_second_part = query_second_half * cos + query_first_half * sin

    key_first_part = key_first_half * cos - key_second_half * sin
    key_second_part = key_second_half * cos + key_first_half * sin

    query_states = torch.cat([query_first_part, query_second_part], dim=-1)
    key_states = torch.cat([key_first_part, key_second_part], dim=-1)

    return query_states, key_states


class PatchTSTNoiser(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        timeseries: torch.Tensor,
        noise_scale: float,
        dim: int | tuple[int, ...] = 1,
    ) -> torch.Tensor:
        """
        Noise the timeseries with standard normal noise

        Parameters:
            timeseries (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Patch input for embedding
            noise_scale (float, *required*):
                Scale of the noise

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`
        """
        noised = timeseries + torch.randn_like(timeseries) * noise_scale
        std = noised.std(dim=dim, keepdim=True)
        std = torch.clamp(std, min=1e-6)
        return noised / std


class PatchTSTClamper(nn.Module):
    """
    This is so stupid
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, low_limit: float, high_limit: float
    ) -> torch.Tensor:
        return torch.clamp(x, min=low_limit, max=high_limit)
        # # ReLU-like clamping that avoids direct comparisons
        # x = x - low_limit
        # x = torch.nn.functional.relu(x)
        # x = torch.nn.functional.relu(high_limit - low_limit - x)
        # x = high_limit - x
        # return x


class PatchTSTFourierApproximator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, timeseries: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use top k modes of the Fourier transform to approximate the timeseries

        Parameters:
            timeseries (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Patch input for embedding
            k (int, *required*):
                Number of modes to use

        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`
        """
        _, seq_length, n_channels = timeseries.shape
        # Vectorized FFT applied on sequence length dimension
        ffts = torch.fft.rfft(timeseries, axis=1)  # Shape: (batch_size, n_freqs, 3)

        # Get indices of top k modes by magnitude
        magnitudes = torch.abs(ffts)
        top_k_indices = torch.argsort(magnitudes, dim=1)[-k:, :]  # Shape: (k, 3)

        # Zero out all but top k modes
        filtered_ffts = torch.zeros_like(ffts)

        for i in range(n_channels):
            filtered_ffts[:, top_k_indices[:, i], i] = ffts[:, top_k_indices[:, i], i]

        # Vectorized inverse transform
        reconstructed = torch.fft.irfft(filtered_ffts, seq_length, dim=1)
        return reconstructed
