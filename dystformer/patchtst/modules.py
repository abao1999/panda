"""
Some modules for PatchTST
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PatchTSTConfig


class PatchTSTKernelEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        assert (
            config.patch_length
            + len(config.poly_degrees) * config.num_poly_feats
            + config.num_rff
            == config.d_model
        ), (
            f"Sum of features must equal d_model: d_poly + d_rff + patch_length = "
            f"{len(config.poly_degrees) * config.num_poly_feats} + {config.num_rff}"
            f" + {config.patch_length} != {config.d_model}"
        )
        self.num_poly_feats = config.num_poly_feats
        self.poly_degrees = config.poly_degrees
        self.patch_indices = [
            torch.randint(
                high=config.patch_length,
                size=(self.num_poly_feats, d),
                requires_grad=False,
            )
            for d in self.poly_degrees
        ]
        self.freq_weights = nn.Parameter(
            config.rff_scale * torch.randn(config.patch_length, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.freq_biases = nn.Parameter(
            torch.randn(1, 1, 1, config.num_rff // 2),
            requires_grad=config.rff_trainable,
        )
        self.projection = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        poly_feats = [x[..., pis].prod(dim=-1) for pis in self.patch_indices]
        weighted_x = x @ self.freq_weights + self.freq_biases
        rff_feats = torch.cat([torch.sin(weighted_x), torch.cos(weighted_x)], dim=-1)
        features = torch.cat([x, *poly_feats, rff_feats], dim=-1)
        return self.projection(features)


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


class LowRankLinear(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        use_rope: bool = True,
        max_wavelength: int = 10000,
        rope_percent: float = 0.5,
        rank: int = 15,
        config: Optional[PatchTSTConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.max_wavelength = max_wavelength
        self.rope_percent = rope_percent
        self.use_rope = use_rope
        self.rank = rank
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Parameters:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`, *required*):
                Patch input for embedding
        """
        bsz, seq_len, dim = hidden_states.shape
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        hidden_heads = self._shape(hidden_states, -1, bsz).reshape(proj_shape)
        U, S, V = torch.linalg.svd(hidden_heads, full_matrices=False)
        V = V[:, : self.rank, :].detach()
        U = U[:, :, : self.rank].detach()
        S = S[:, : self.rank].detach()

        V = S[..., None] * self.v_proj(V)
        lowrank_output = torch.bmm(U, V)
        lowrank_output = lowrank_output.view(
            bsz, self.num_heads, seq_len, self.head_dim
        )
        lowrank_output = lowrank_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        lowrank_output = lowrank_output.reshape(bsz, seq_len, self.embed_dim)

        lowrank_output = self.out_proj(lowrank_output)

        return lowrank_output, None, None


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

        batch_size, seq_length, n_channels = timeseries.shape
        # Vectorized FFT applied on sequence length dimension
        ffts = torch.fft.rfft(timeseries, axis=1)  # Shape: (batch_size, n_freqs, 3)
        # Get indices of top k modes by magnitude
        magnitudes = torch.abs(ffts)
        # Shape: (batch_size, k, 3)
        top_k_indices = torch.argsort(magnitudes, dim=1)[:, -k:, :]
        # Zero out all but top k modes
        filtered_ffts = torch.zeros_like(ffts)

        for b in range(batch_size):
            for i in range(n_channels):
                filtered_ffts[b, top_k_indices[b, :, i], i] = ffts[
                    b, top_k_indices[b, :, i], i
                ]

        # Vectorized inverse transform
        reconstructed = torch.fft.irfft(filtered_ffts, seq_length, dim=1)
        return reconstructed
