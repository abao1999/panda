# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import itertools
from typing import Callable, Generator, Iterator

import numpy as np
import torch
from gluonts.itertools import Cyclic, Filter
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    FilterTransformation,
    InstanceSplitter,
    LeavesMissingValues,
    MissingValueImputation,
    TestSplitSampler,
    ValidationSplitSampler,
)
from torch.utils.data import IterableDataset, get_worker_info

from dystformer.chronos.tokenizer import ChronosTokenizer


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class RestarableIteratorWrapper:
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
        self._length = None

    def __iter__(self):
        yield from self.generator_func(*self.args, **self.kwargs)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series
    into a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and
    ``labels``.

    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: list[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: int | None = None,
        model_type: str = "seq2seq",
        imputation_method: MissingValueImputation | None = None,
        mode: str = "train",
        np_dtype: np.dtype = np.dtype(np.float32),
        augmentations: list[Callable] | None = None,
        augmentation_probabilities: list[float] | None = None,
        transforms: list[Callable] | None = None,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("train", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype
        self.transforms = transforms
        self.augmentations = augmentations
        self.augmentation_probabilities = augmentation_probabilities

        if self.augmentations is None:
            return

        if self.augmentation_probabilities is None:
            self.augmentation_probabilities = [1.0 / len(self.augmentations)] * len(
                self.augmentations
            )

        assert len(self.augmentations) == len(self.augmentation_probabilities)
        assert sum(self.augmentation_probabilities) == 1.0

    def preprocess_iter(self, entry: Filter, mode: str) -> Generator[dict, None, None]:
        for item in entry:
            target = np.asarray(item["target"], dtype=self.np_dtype)

            if mode == "train" and self.augmentations is not None:
                augmentation_idx = np.random.choice(
                    len(self.augmentations), p=self.augmentation_probabilities
                )
                target = self.augmentations[augmentation_idx](target)

            for transform in self.transforms or []:
                target = transform(target)

            for i in range(target.shape[0]):
                univariate_target = target[i]

                if self.model_type == "causal":
                    univariate_target = self.imputation_method(univariate_target)

                if mode == "train" and self.drop_prob > 0:
                    mask = np.random.choice(
                        [True, False],
                        size=len(univariate_target),
                        p=[self.drop_prob, 1 - self.drop_prob],
                    )
                    univariate_target = np.where(mask, np.nan, univariate_target)

                yield {"start": item["start"], "target": univariate_target}

    def _create_instance_splitter(self, mode: str):
        assert mode in ["train", "test", "validation"]

        instance_sampler = {
            "train": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "train"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts

            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            # padded_input_ids, obs_input_ids = torch.tensor_split(
            #     input_ids_tensor, [pad_start_idx], dim=-1
            # )
            # padded_attention_mask, obs_attention_mask = torch.tensor_split(
            #     attention_mask, [pad_start_idx], dim=-1
            # )
            input_ids_tensor = torch.tensor(input_ids)
            padded_input_ids = input_ids_tensor[:pad_start_idx]
            obs_input_ids = input_ids_tensor[pad_start_idx:]
            padded_attention_mask = attention_mask[:, :pad_start_idx]
            obs_attention_mask = attention_mask[:, pad_start_idx:]

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                dim=-1,  # Changed 'axis' to 'dim'
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                dim=-1,  # Changed 'axis' to 'dim'
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            RestarableIteratorWrapper(self.preprocess_iter, dataset, self.mode)
            for dataset in self.datasets
        ]

        if self.mode == "train":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]  # each iterable is cycle iterator over the individual channels for a trajectory
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "train":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    # iterators[idx] is a cycler over the individual channels for a trajectory
                    # calling next pulls random channel/dim from the sampled trajectory (idx)
                    # the fact that iterators[idx] is a cycler ensures each dim is seen with sufficient iterations
                    data = next(iterators[idx])
                    yield self.to_hf_format(data)
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            # cyclers aren't used here, so just chain iterators sequentially
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)
