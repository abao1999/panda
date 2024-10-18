"""
PatchTST model with the forward pass exposed

TODO: This whole thing is kinda useless, figure out how to gracefully deprecate
"""

import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn

from dystformer.patchtst.patchtst import (
    PatchTSTConfig,
    PatchTSTForPrediction,
    PatchTSTForPretraining,
)


def left_pad_and_stack_multivariate(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Left pad a list of multivariate time series tensors to the same length and stack them.
    Used in pipeline, if given context is a list of tensors.
    """
    max_len = max(c.shape[0] for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 2
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


class PatchTST(nn.Module):
    def __init__(
        self,
        config: dict,
        mode: str = "predict",
        pretrained_encoder_path: Optional[str] = None,
    ):
        super().__init__()

        self.mode = mode
        assert mode in [
            "pretrain",
            "predict",
        ], "Mode must be either 'pretrain' or 'predict'"

        # Convert the config dict to PatchTSTConfig
        self.config = PatchTSTConfig(**config)

        if mode == "pretrain":
            self.model = PatchTSTForPretraining(self.config)
        elif mode == "predict":
            self.model = PatchTSTForPrediction(self.config)

        if pretrained_encoder_path is not None:
            self.load_pretrained_encoder(pretrained_encoder_path)

    @property
    def device(self):
        return self.model.device

    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)

    def load_pretrained_encoder(self, pretrain_path: str):
        """
        Load a pretrained encoder from a PatchTSTForPretraining model and replace the current encoder.

        Parameters:
            pretrain_path (str): Path to the directory containing the pretrained PatchTSTForPretraining model.
        """
        # Ensure the method is only called in "predict" mode
        if self.mode != "predict":
            raise RuntimeError(
                "load_pretrained_encoder can only be called in 'predict' mode."
            )

        pretrained_model = PatchTSTForPretraining.from_pretrained(pretrain_path)

        # Replace the current encoder with the pretrained encoder
        self.model.model.encoder = pretrained_model.model.encoder

    @classmethod
    def from_pretrained(cls, pretrain_path: str, **kwargs):
        """
        Load a pretrained model from a path.
        """
        config = PatchTSTConfig.from_pretrained(pretrain_path)
        model = cls(config=config.to_dict(), mode=kwargs.get("mode", "predict"))
        model.model = PatchTSTForPretraining.from_pretrained(pretrain_path)
        return model

    def _prepare_and_validate_context(
        self, context: Union[torch.Tensor, List[torch.Tensor]]
    ):
        if isinstance(context, list):
            context = left_pad_and_stack_multivariate(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.view(1, -1, 1)
        assert context.ndim == 3

        return context

    @torch.no_grad()
    def predict(
        self,
        context: Union[torch.Tensor, List[torch.Tensor]],
        prediction_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        limit_prediction_length: bool = True,
    ) -> torch.Tensor:
        """
        Get forecasts for the given time series.

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to what specified
            in ``self.model.config``.
        num_samples
            Number of sample paths to predict. Defaults to what
            specified in ``self.model.config``.
        temperature
            Temperature to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_k
            Top-k parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        top_p
            Top-p parameter to use for generating sample tokens.
            Defaults to what specified in ``self.model.config``.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. True by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.

        Returns
        -------
        samples
            Tensor of sample forecasts, of shape
            (batch_size, num_samples, prediction_length).
        """
        assert (
            self.mode == "predict"
        ), "Model must be in predict mode to use this method"

        context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length is None:
            prediction_length = self.config.prediction_length

        if prediction_length > self.model.config.prediction_length:
            msg = (
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )
            if limit_prediction_length:
                msg += "You can turn off this check by setting `limit_prediction_length=False`."
                raise ValueError(msg)
            warnings.warn(msg)

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            # prediction: [bs x num_samples x forecast_len x num_channels]
            outputs = self.model.generate(context_tensor)
            prediction = outputs.sequences  # type: ignore

            predictions.append(prediction)
            remaining -= prediction.shape[-2]

            if remaining <= 0:
                break

            # need to contract over the num_samples dimension, use median
            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=-1
            )

        # shape: [bs x num_samples x prediction_length x num_channels]
        return torch.cat(predictions, dim=-1)

    def forward(self, *args, **kwargs):
        """
        Exposed for flexibility in channel mixing strategies.
        """
        # pretraining mode does not require forecasts
        if self.mode == "pretrain":
            kwargs.pop("future_values", None)

        return self.model(*args, **kwargs)
