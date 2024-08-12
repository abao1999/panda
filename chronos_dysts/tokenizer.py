# Adapted from https://github.com/amazon-science/chronos-forecasting
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: modify for dysts

import torch
from typing import Any, Optional, Tuple
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional
import importlib


@dataclass
class ChronosConfig:
    """
    This class holds all the configuration parameters to be used
    by ``ChronosTokenizer`` and ``ChronosModel``.
    """

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    context_length: int
    prediction_length: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer": # this type hint is source of circular import
        # # can also move to end of file and use direct class registry or factory function
        class_ = globals().get(self.tokenizer_class) # solution with globals, not recommended
        # module = importlib.import_module(__name__) # __import__(__name__)
        # class_ = getattr(module, self.tokenizer_class)
        return class_(**self.tokenizer_kwargs, config=self)
    


class ChronosTokenizer:
    """
    A ``ChronosTokenizer`` definines how time series are mapped into token IDs
    and back.

    For details, see the ``input_transform`` and ``output_transform`` methods,
    which concrete classes must implement.
    """

    def context_input_transform(
        self,
        context: torch.Tensor,
    ) -> Tuple:
        """
        Turn a batch of time series into token IDs, attention map, and tokenizer_state.

        Parameters
        ----------
        context
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        tokenizer_state
            An object that can be passed to ``label_input_transform``
            and ``output_transform``. Contains the relevant information
            to decode output samples into real values,
            such as location and scale parameters.
        """
        raise NotImplementedError()

    def label_input_transform(self, label: torch.Tensor, tokenizer_state: Any) -> Tuple:
        """
        Turn a batch of label slices of time series into token IDs and attention map
        using the ``tokenizer_state`` provided by ``context_input_transform``.

        Parameters
        ----------
        context
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.
        tokenizer_state
            An object returned by ``context_input_transform`` containing
            relevant information to preprocess data, such as location and
            scale. The nature of this depends on the specific tokenizer.
            This is used for tokenizing the label, in order to use the same
            scaling used to tokenize the context.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        """
        raise NotImplementedError()

    def output_transform(
        self, samples: torch.Tensor, tokenizer_state: Any
    ) -> torch.Tensor:
        """
        Turn a batch of sample token IDs into real values.

        Parameters
        ----------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing token IDs of sample trajectories.
        tokenizer_state
            An object returned by ``input_transform`` containing
            relevant context to decode samples, such as location and scale.
            The nature of this depends on the specific tokenizer.

        Returns
        -------
        forecasts
            A real tensor, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(
        self, low_limit: float, high_limit: float, config: ChronosConfig
    ) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def _input_transform(
        self, context: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        return token_ids, attention_mask, scale

    def _append_eos_token(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
        token_ids = torch.concat((token_ids, eos_tokens), dim=1)
        eos_mask = torch.full((batch_size, 1), fill_value=True)
        attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask

    def context_input_transform(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, attention_mask, scale = self._input_transform(context=context)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask, scale

    def label_input_transform(
        self, label: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        length = label.shape[-1]

        assert length == self.config.prediction_length
        token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

        if self.config.use_eos_token:
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask

    def output_transform(
        self, samples: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed