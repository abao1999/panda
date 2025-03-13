import random
from itertools import accumulate

import hydra
import torch

from dystformer.patchtst.pipeline import FixedSubsetChannelSampler
from dystformer.utils import load_patchtst_model


def create_block_attention_mask(
    num_channels: list[int], expand_dim: int = 1
) -> torch.Tensor:
    mask = torch.zeros(sum(num_channels), sum(num_channels)).bool()
    for origin, width in zip(accumulate([0] + num_channels), num_channels):
        mask[origin : origin + width, origin : origin + width] = True

    attn_mask = torch.zeros_like(mask, dtype=torch.float)
    attn_mask.masked_fill_(~mask, float("-inf"))
    return attn_mask.view(1, 1, *mask.shape).repeat(expand_dim, 1, 1, 1)


def test_prediction_basic(model, cfg):
    """
    Test heterogeneous num_channels and autoregressive continuing context prediction
    """
    x = torch.randn(4, 1034, 3).to("cuda")
    y = torch.randn(7, 853, 7).to("cuda")

    # test forward pass works for abitrary num channels
    assert model(x).prediction_outputs.shape == torch.Size(
        [4, cfg.patchtst.prediction_length, 3]
    )
    assert model(y).prediction_outputs.shape == torch.Size(
        [7, cfg.patchtst.prediction_length, 7]
    )

    # test prediction with continuing context
    context = x
    for i in range(10):
        outputs = model(context)
        context = torch.cat([context, outputs.prediction_outputs], dim=1)

    assert context.shape == torch.Size(
        [4, x.size(1) + 10 * cfg.patchtst.prediction_length, 3]
    )

    # test predict method with a channel sampler for handling a batch of heterogeneous num_channels
    num_samples = 2
    num_channels = 3
    x = torch.rand(512, 5).to("cuda")
    y = torch.rand(512, 7).to("cuda")
    z = torch.rand(512, 4).to("cuda")
    tensor_list = [x, y, z]
    sampler = FixedSubsetChannelSampler(
        num_channels=num_channels, num_samples=num_samples
    )
    prediction = model.predict(tensor_list, channel_sampler=sampler)
    total_samples = sum(
        num_samples * (c - num_channels + 1) for c in [d.shape[-1] for d in tensor_list]
    )
    assert prediction.shape == torch.Size(
        [total_samples, 1, cfg.patchtst.prediction_length, num_channels]
    )


def test_attn_mask(batch_size: int, model, cfg):
    """
    Test block attention mask for handling a batch with heterogeneous num_channels
    """
    num_channels = [random.randint(3, 6) for _ in range(batch_size)]
    batch = [torch.randn(512, nc).float() for nc in num_channels]

    # forward pass the batch via a for loop
    outputs_looped = torch.cat(
        [model(b[None, ...].to("cuda")).prediction_outputs for b in batch], dim=-1
    )

    # forward pass the batch via a single forward pass
    batch_tensor = torch.cat(batch, dim=1).unsqueeze(0).to("cuda")
    block_attention_mask = create_block_attention_mask(
        num_channels, expand_dim=batch_tensor.size(1) // cfg.patchtst.patch_length
    )
    outputs_batched = model(
        batch_tensor, channel_attention_mask=block_attention_mask
    ).prediction_outputs

    assert torch.allclose(outputs_looped, outputs_batched, rtol=1e-3, atol=1e-7)

    # in case the allclose is not convincing
    print(
        "Max absolute difference:",
        torch.abs(outputs_looped - outputs_batched).max().item(),
    )


def test_interleaved_prediction(
    batch_size: int, interleave_dim: int, num_interleave_samples: int, model, cfg
):
    """
    Test interleaved prediction
    """
    num_channels = [random.randint(3, 10) for _ in range(batch_size)]
    channel_offsets = [o for o in accumulate([0] + num_channels[:-1])]
    indices = [torch.arange(nc) for nc in num_channels]

    batch = [torch.randn(512, nc).float() for nc in num_channels]
    batch_tensor = torch.cat(batch, dim=1).unsqueeze(0).to("cuda")

    def sample_indices(
        inds: torch.Tensor, sample_size: int, num_samples: int, num_oversample: int = 2
    ) -> torch.Tensor:
        """Sample index subsets of size sample_size from inds"""
        index_subsets = torch.stack(
            [inds[torch.randperm(len(inds))[:sample_size]] for _ in range(num_samples)]
        )

        # ensure that the ind subsets cover all the indices
        oversampled = 0
        indset = set(inds.flatten().tolist())
        while (
            not set(index_subsets.flatten().tolist()) == indset
        ) or oversampled < num_oversample:
            sample = inds[torch.randperm(len(inds))[:sample_size]]
            index_subsets = torch.cat([index_subsets, sample[None, ...]], dim=0)
            oversampled += 1

        return index_subsets

    interleave_indices = [
        sample_indices(inds, interleave_dim, num_interleave_samples) for inds in indices
    ]
    sample_offsets = [o for o in accumulate([0] + list(map(len, interleave_indices)))]
    shifted_interleave_indices = torch.cat(
        [(idx + offset) for idx, offset in zip(interleave_indices, channel_offsets)],
        dim=0,
    )

    # index the subsets from the channels and move the index samples to the batch dimension
    subsets = batch_tensor[..., shifted_interleave_indices]
    bs, T, nsamples, _ = subsets.shape
    subsets = subsets.transpose(1, 2).view(bs * nsamples, T, -1)

    # [L, bs * nsamples, nc]
    outputs = model(subsets).prediction_outputs.transpose(0, 1)
    L = outputs.size(0)

    # assuming bs == 1
    # NAIVE method
    naive_results = []
    for i, nc in enumerate(num_channels):
        inds = interleave_indices[i]
        preds = (
            outputs[:, sample_offsets[i] : sample_offsets[i] + len(inds), :]
            .view(-1, interleave_dim * len(inds))
            .to("cpu")
        )
        result = torch.zeros(L, nc)
        for j, ind in enumerate(inds.flatten()):
            result[:, ind] += preds[:, j]
        _, cnts = torch.unique(inds, return_counts=True)
        result = result / cnts[None, :]
        naive_results.append(result)

    # scatter method
    scatter_results = []
    for i, nc in enumerate(num_channels):
        inds = interleave_indices[i]
        preds = (
            outputs[:, sample_offsets[i] : sample_offsets[i] + len(inds), :]
            .view(-1, interleave_dim * len(inds))
            .to("cpu")
        )
        inds_flat = inds.flatten().unsqueeze(0)
        result = torch.zeros(L, nc).scatter_reduce_(
            1, inds_flat.repeat(L, 1), preds, reduce="mean", include_self=False
        )
        scatter_results.append(result)

    assert all(
        torch.allclose(naive_results[i], scatter_results[i], rtol=1e-3, atol=1e-7)
        for i in range(len(num_channels))
    )


def test_model_fwd(cfg):
    """
    Test the forward pass of the model
    """
    model = load_patchtst_model(
        mode="predict",
        model_config=dict(cfg.patchtst),
        pretrained_encoder_path=cfg.patchtst.pretrained_encoder_path,
    )
    past_values = torch.randn(10, 512, 3)
    future_values = torch.randn(10, 512, 3)
    model(past_values=past_values, future_values=future_values)
    breakpoint()


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # model = PatchTSTPipeline.from_pretrained(
    #     mode="predict", pretrain_path=cfg.patchtst.pretrain_path
    # )
    test_model_fwd(cfg)

    # test_prediction_basic(model, cfg)
    # test_attn_mask(batch_size=3, model=model, cfg=cfg)
    # test_interleaved_prediction(
    #     batch_size=5,
    #     interleave_dim=3,
    #     num_interleave_samples=2,
    #     model=model,
    #     cfg=cfg,
    # )


if __name__ == "__main__":
    main()
