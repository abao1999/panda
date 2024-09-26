# Adapted from https://github.com/amazon-science/chronos-forecasting
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    GenerationConfig,
    PreTrainedModel,
)

from dystformer.chronos.tokenizer import ChronosConfig


class ChronosModel(nn.Module):
    """
    A ``ChronosModel`` wraps a ``PreTrainedModel`` object from ``transformers``
    and uses it to predict sample paths for time series tokens.

    Parameters
    ----------
    config
        The configuration to use.
    model
        The pretrained model to use.
    """

    def __init__(self, config: ChronosConfig, model: PreTrainedModel) -> None:
        super().__init__()
        self.config = config
        self.model = model

    @property
    def device(self):
        return self.model.device

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Extract the encoder embedding for the given token sequences.

        Parameters
        ----------
        input_ids
            Tensor of indices of input sequence tokens in the vocabulary
            with shape (batch_size, sequence_length).
        attention_mask
            A mask tensor of the same shape as input_ids to avoid attending
            on padding or missing tokens.

        Returns
        -------
        embedding
            A tensor of encoder embeddings with shape
            (batch_size, sequence_length, d_model).
        """
        assert (
            self.config.model_type == "seq2seq"
        ), "Encoder embeddings are only supported for encoder-decoder models"
        return self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Predict future sample tokens for the given token sequences.

        Arguments ``prediction_length``, ``num_samples``, ``temperature``,
        ``top_k``, ``top_p`` can be used to customize the model inference,
        and default to the corresponding attributes in ``self.config`` if
        not provided.

        Returns
        -------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        if num_samples is None:
            num_samples = self.config.num_samples
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k
        if top_p is None:
            top_p = self.config.top_p

        preds = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(
                min_new_tokens=prediction_length,
                max_new_tokens=prediction_length,
                do_sample=True,
                num_return_sequences=num_samples,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            ),
        )

        if self.config.model_type == "seq2seq":
            preds = preds[..., 1:]  # remove the decoder start token
        else:
            assert self.config.model_type == "causal"
            assert preds.size(-1) == input_ids.size(-1) + prediction_length  # type: ignore
            preds = preds[..., -prediction_length:]

        return preds.reshape(input_ids.size(0), num_samples, -1)
