"""
Exposed PatchTST model, taken from HuggingFace transformers
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTPreTrainedModel
from transformers.models.patchtst.modeling_patchtst import (
    BaseModelOutput,
    PatchTSTAttention,
    PatchTSTEncoderLayer,
    PatchTSTForPretrainingOutput,
    PatchTSTMasking,
    PatchTSTMaskPretrainHead,
    PatchTSTModelOutput,
    PatchTSTPatchify,
    PatchTSTPositionalEncoding,
    PatchTSTScaler,
)


class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.num_input_channels = config.num_input_channels
        # Input encoding: projection of feature vectors onto a d-dim vector space
        self.input_embedding = nn.Linear(config.patch_length, config.d_model)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        # Input encoding
        embeddings = self.input_embedding(
            patch_input
        )  # x: [bs x num_channels  x num_patches x d_model]
        return embeddings


class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """

    def __init__(self, config: PatchTSTConfig, num_patches: int):
        super().__init__(config)
        self.gradient_checkpointing = False

        # Input embedding: projection of feature vectors onto a d-dim vector space
        self.embedder = PatchTSTEmbedding(config)
        # Positional encoding
        self.positional_encoder = PatchTSTPositionalEncoding(config, num_patches)
        # Encoder
        self.layers = nn.ModuleList(
            [PatchTSTEncoderLayer(config) for i in range(config.num_hidden_layers)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        patch_input: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BaseModelOutput:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Past values of the time series
            output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
            output_attentions (bool, optional): Indicates if attentions should be outputted.

        return:
            `BaseModelOutput`
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Input embedding
        patch_input = self.embedder(patch_input)
        # Positional encoding
        hidden_state = self.positional_encoder(patch_input)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_state,)  # type: ignore

            layer_outputs = encoder_layer(
                hidden_state=hidden_state, output_attentions=output_attentions
            )
            # get hidden state. hidden_state shape is [bs x num_channels x num_patches x d_model]
            # or [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            hidden_state = layer_outputs[0]
            # append attention matrix at each layer
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # type: ignore
        # return past_values, hidden_states
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        self.do_mask_input = config.do_mask_input
        # get num_patches information from PatchTSTPatchify
        num_patches = self.patchifier.num_patches

        if self.do_mask_input:
            self.masking = PatchTSTMasking(config)
        else:
            self.masking = nn.Identity()
        self.encoder = PatchTSTEncoder(config, num_patches=num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTModelOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.BoolTensor` of shape `(batch_size, prediction_length, num_input_channels)`, *optional*):
                Future target values associated with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*):
                Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTModelOutput` or tuple of `torch.Tensor` (if `return_dict`=False or `config.return_dict`=False)

        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # x: tensor [bs x sequence_length x num_input_channels]
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        # patched_values: [bs x num_input_channels x num_patches x patch_length] for pretrain
        patched_values = self.patchifier(scaled_past_values)
        if self.do_mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = self.masking(patched_values), None

        encoder_output = self.encoder(
            patch_input=masked_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if not return_dict:
            outputs = (
                encoder_output.last_hidden_state,
                encoder_output.hidden_states,
                encoder_output.attentions,
            )
            outputs = outputs + (mask, loc, scale, patched_values)
            return tuple(v for v in outputs if v is not None)

        return PatchTSTModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            mask=mask,  # type: ignore
            loc=loc,
            scale=scale,
            patch_input=patched_values,
        )


class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.do_mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = PatchTSTMaskPretrainHead(config)
        self.loss = nn.MSELoss(reduction="none")

        if config.mix_channels:
            self.mixing_head = PatchTSTAttention(
                embed_dim=config.context_length,
                num_heads=config.num_mixing_heads,
                dropout=config.mixing_dropout,
                is_decoder=False,
                bias=False,
                is_causal=False,
            )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTForPretrainingOutput]:
        r"""
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            output_attentions (`bool`, *optional*):
                Whether or not to return the output attention of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPretrainingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # past_values: [bs x num_channels x num_patches x d_model] or
        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # last_hidden_state: [bs x num_channels x num_patches x d_model] or
        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        # x_hat: [bs x num_channels x num_patches x patch_length]
        x_hat = self.head(model_output.last_hidden_state)

        # self attention over the channels (permutation invariantly)
        if self.config.mix_channels:
            x_hat, attn_weights, past_kv = self.mixing_head(
                x_hat.view(*x_hat.shape[:2], -1),
                output_attentions=output_attentions,
            )  # x_hat: [bs x num_channels x num_patches x d_model]

        # calculate masked_loss
        loss_val = self.loss(x_hat, model_output.patch_input)

        # reduce over the patch length dim first, then compute the masked loss over the tokens
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (
            model_output.mask.sum() + 1e-10
        )

        encoder_states = model_output.hidden_states
        if not return_dict:
            outputs = (x_hat,) + model_output[1:-4]
            outputs = (masked_loss,) + outputs if masked_loss is not None else outputs
            return outputs
        return PatchTSTForPretrainingOutput(
            loss=masked_loss,
            prediction_output=x_hat,
            hidden_states=encoder_states,
            attentions=model_output.attentions,
        )
