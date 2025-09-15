"""
PatchTST model wrapper for the forecaster and masked model

TODO: This whole thing is kinda useless, figure out how to gracefully deprecate
"""

import warnings
from dataclasses import dataclass

import torch

from panda.patchtst.patchtst import (
    PatchTSTForPrediction,
    PatchTSTForPretraining,
)
from panda.utils import left_pad_and_stack_multivariate


@dataclass
class PatchTSTPipeline:
    """
    PatchTST pipeline for inference
    """

    mode: str
    model: PatchTSTForPretraining | PatchTSTForPrediction

    @property
    def device(self):
        return self.model.device

    @classmethod
    def from_pretrained(cls, mode: str, pretrain_path: str, **kwargs):
        """
        Load a pretrained model from a path and move it to the specified device.
        """
        model_class = {
            "pretrain": PatchTSTForPretraining,
            "predict": PatchTSTForPrediction,
        }[mode]
        model = model_class.from_pretrained(pretrain_path, **kwargs)
        return cls(mode=mode, model=model)

    def _prepare_and_validate_context(
        self, context: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(context, list):
            assert len(set(c.shape[-1] for c in context)) == 1, (
                "All contexts must have the same number of channels"
                "Use a channel sampler to subsample a fixed number of channels"
            )
            context = left_pad_and_stack_multivariate(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.view(1, -1, 1)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        assert context.ndim == 3

        return context.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int,
        limit_prediction_length: bool = True,
        sliding_context: bool = False,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate an autoregressive forecast for a given context timeseries

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is sequence length. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to what specified
            in ``self.model.config``.
        limit_prediction_length
            Force prediction length smaller or equal than the
            built-in prediction length from the model. True by
            default. When true, fail loudly if longer predictions
            are requested, otherwise longer predictions are allowed.
        sliding_context
            If True, the context window will be slid over the time series, otherwise
            the context window will be accumulated and grows in memory.

        Returns
        -------
        samples
            Tensor of sample forecasts, of shape
            [bs x num_samples x prediction_length x num_channels]
        """
        assert self.mode == "predict", (
            "Model must be in predict mode to use this method"
        )

        # context_tensor: [bs x context_length x num_channels]
        context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length > self.model.config.prediction_length and verbose:
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
            outputs = self.model.generate(context_tensor)

            # prediction: [bs x num_samples x forecast_len x num_channels]
            prediction = outputs.sequences  # type: ignore

            predictions.append(prediction)
            remaining -= prediction.shape[2]

            if remaining <= 0:
                break

            # need to contract over the num_samples dimension, use median
            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=1
            )

            # dont grow the context window, only keep the most recent context_length
            if sliding_context:
                context_tensor = context_tensor[
                    :, -self.model.config.context_length :, :
                ]

        # shape: [bs x num_samples x prediction_length x num_channels]
        predictions = torch.cat(predictions, dim=2)

        return predictions

    @torch.no_grad()
    def complete(
        self,
        context: torch.Tensor | list[torch.Tensor],
        past_observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get completions for the given time series.
        TODO: do autoregressive completion / stitching together of completions

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is batch. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.

        Returns
        -------
        completions
            Tensor of completions, of shape
            [bs x context_length x num_channels]
        """
        assert self.mode == "pretrain", (
            "Model must be in pretrain mode to use this method"
        )

        context_tensor = self._prepare_and_validate_context(context=context)
        completions_output = self.model.generate_completions(
            context_tensor, past_observed_mask=past_observed_mask
        )
        # TODO: need to check shapes
        completions = completions_output.completions.view_as(context_tensor).permute(
            0, 2, 1
        )
        loc = completions_output.loc
        scale = completions_output.scale
        # mask = completions_output.mask
        # unod the instance normalization
        completions = loc + scale * completions
        return completions
