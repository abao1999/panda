"""
Unified datasets for Panda
Includes:
- UnivariateTimeSeriesDataset (Chronos-style, tokenizer-based)
- MultivariateTimeSeriesDataset (PatchTST-style, raw tensors)
Shared samplers and shuffle utilities are defined here.
"""

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cyclic, Map
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    FilterTransformation,
    InstanceSampler,
    InstanceSplitter,
    LeavesMissingValues,
    MissingValueImputation,
)
from torch.utils.data import IterableDataset, get_worker_info

from panda.chronos.model import ChronosTokenizer

# used for prediction length in test mode when window style is single
# if you're predicting for more timepoints than this at a time...what are you doing??
MAX_PREDICTION_LENGTH = 1_000_000


class RegularWindowedSampler(InstanceSampler):
    """
    Sample regular context windows from each series.

    Parameters
    ----------
    stride: int
        stride of the sampled context windows
    """

    stride: int

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)

        return np.arange(a, b + 1, self.stride)


class SingleContextSampler(InstanceSampler):
    """
    Sample a single context window from a specified start, or the beginning of the series

    Used for autoregressive prediction where the model should predict the
    rest of the entire timeseries.
    """

    start: int | None = None

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)
        if self.start is None:
            return np.array([a])
        if self.start < a or self.start > b:
            return np.array([], dtype=int)

        return np.array([self.start])


class NumInstanceSampler(InstanceSampler):
    """
    Samples N time points from each series.

    Parameters
    ----------
    N
        number of time points to sample from each time series.
    """

    N: int
    rng: np.random.Generator

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)

        return self.rng.integers(a, b + 1, size=self.N)


class PseudoShuffledIterableDataset(IterableDataset):
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
                idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class BaseTimeSeriesDataset(IterableDataset, ShuffleMixin, ABC):
    """Common windowing, splitting, and iteration logic for datasets."""

    context_length: int
    prediction_length: int
    window_style: str
    mode: str
    probabilities: list[float]

    # window sampling arguments
    window_start: int | None = None
    window_stride: int | None = None
    num_test_instances: int | None = None

    @property
    @abstractmethod
    def rng(self) -> np.random.Generator: ...

    @abstractmethod
    def _preprocessed_datasets(self) -> list: ...

    @abstractmethod
    def _format_train(self, entry: dict) -> dict: ...

    @abstractmethod
    def _format_eval(self, entry: dict) -> dict: ...

    def _create_instance_splitter(self, mode: str):
        assert mode in ["train", "test"]
        assert self.window_style in ["sampled", "rolling", "single"]

        if self.window_stride == "rolling":
            assert self.window_stride and self.window_stride > 0
        elif self.window_style == "sampled":
            assert self.num_test_instances and self.num_test_instances > 0

        window_sampler = {
            "sampled": partial(NumInstanceSampler, N=self.num_test_instances, rng=self.rng),
            "rolling": partial(RegularWindowedSampler, stride=self.window_stride),
            "single": partial(SingleContextSampler, start=self.window_start),
        }[self.window_style]

        instance_sampler = {
            "train": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.context_length,
                min_future=self.prediction_length,
            ),
            "test": window_sampler(min_past=self.context_length, min_future=self.prediction_length),
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
        split_transform = self._create_instance_splitter("train") + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        return split_transform.apply(data, is_train=True)

    def create_test_data(self, data):
        return self._create_instance_splitter("test").apply(data, is_train=False)

    def __iter__(self) -> Iterator:
        iterables = self._preprocessed_datasets()

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(itertools.islice(self.probabilities, worker_id, None, num_workers))

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "train":
            while True:
                idx = int(self.rng.choice(range(len(iterators)), p=probs))
                try:
                    entry = next(iterators[idx])
                    yield self._format_train(entry)
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self._format_eval(entry)


class RestartableIterator:
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
        self._length = None

    def __iter__(self):
        yield from self.generator_func(*self.args, **self.kwargs)


@dataclass
class UnivariateTimeSeriesDataset(BaseTimeSeriesDataset):
    datasets: list
    probabilities: list[float]
    tokenizer: ChronosTokenizer | None = None
    patch_size: int | None = None
    context_length: int = 512
    prediction_length: int = 64
    drop_prob: float = 0.2
    min_past: int | None = None
    model_type: str = "seq2seq"
    imputation_method: MissingValueImputation | None = None
    mode: str = "train"
    np_dtype: np.dtype = np.dtype(np.float32)
    num_test_instances: int = 1
    window_style: str = "sampled"
    window_start: int = 0
    window_stride: int = 1
    transforms: list[Callable] | None = None
    augmentations: list[Callable] | None = None
    augmentation_probabilities: list[float] | None = None
    augmentation_rate: float = 0.0
    random_seed: int = 1337

    # add a single dim at a specified dim
    singleton: int | None = None

    def __post_init__(self):
        super().__init__()
        assert len(self.probabilities) == len(self.datasets)
        assert self.mode in ("train", "test")
        assert self.model_type in ("seq2seq", "causal")
        self.drop_prob = self.drop_prob if self.model_type == "seq2seq" else 0.0
        self.min_past = self.min_past or self.prediction_length
        self.imputation_method = self.imputation_method or LeavesMissingValues()
        self._rng = np.random.default_rng(self.random_seed)
        self.splitter = self._create_instance_splitter(self.mode)

        if self.augmentations is not None:
            if self.augmentation_probabilities is None:
                self.augmentation_probabilities = [1.0 / len(self.augmentations)] * len(self.augmentations)
            assert len(self.augmentations) == len(self.augmentation_probabilities)
            assert sum(self.augmentation_probabilities) == 1.0

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def preprocess_iter(self, dataset: Dataset, mode: str) -> Iterator[dict]:
        for entry in dataset:
            target = np.asarray(entry["target"], dtype=self.np_dtype)

            if mode == "train" and self.augmentations is not None:
                if self.rng.random() < self.augmentation_rate:
                    augmentation_idx = self.rng.choice(len(self.augmentations), p=self.augmentation_probabilities)
                    target = self.augmentations[augmentation_idx](target)

            for transform in self.transforms or []:
                target = transform(target)

            if mode == "train" and self.drop_prob > 0:
                mask = self.rng.choice(
                    [True, False],
                    size=len(target),
                    p=[self.drop_prob, 1 - self.drop_prob],
                )
                target = np.where(mask, np.nan, target)

            prepro_entry = {"start": entry["start"], "target": target}
            for prepro_item in self.splitter.apply([prepro_entry], is_train=self.mode == "train"):
                past_target = prepro_item["past_target"]
                future_target = prepro_item["future_target"]

                for i in range(past_target.shape[1]):
                    univariate_past = past_target[:, i]  # shape: (context_length,)
                    univariate_future = future_target[:, i]  # shape: (prediction_length,)
                    univariate_entry = {"past_target": univariate_past, "future_target": univariate_future}

                    if self.mode == "train":
                        univariate_entry["past_is_pad"] = prepro_item["past_is_pad"]

                    yield univariate_entry

    def _preprocessed_datasets(self) -> list:
        return [RestartableIterator(self.preprocess_iter, dataset, self.mode) for dataset in self.datasets]

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(  # type: ignore
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)  # type: ignore
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            input_ids_tensor = torch.tensor(input_ids)
            padded_input_ids = input_ids_tensor[:pad_start_idx]
            obs_input_ids = input_ids_tensor[pad_start_idx:]
            padded_attention_mask = attention_mask[:, :pad_start_idx]
            obs_attention_mask = attention_mask[:, pad_start_idx:]

            input_ids = torch.cat([obs_input_ids, labels, padded_input_ids], dim=-1)
            attention_mask = torch.cat([obs_attention_mask, labels_mask, padded_attention_mask], dim=-1)
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id  # type: ignore
            labels[~attention_mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def to_hf_format_eval(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"])
        future_target = torch.tensor(entry["future_target"])
        if self.singleton:
            past_target = past_target.unsqueeze(self.singleton)
            future_target = future_target.unsqueeze(self.singleton)
        return {
            "past_values": past_target,
            "future_values": future_target,
        }

    def _format_train(self, entry: dict) -> dict:
        return self.to_hf_format(entry)

    def _format_eval(self, entry: dict) -> dict:
        return self.to_hf_format_eval(entry)


@dataclass
class MultivariateTimeSeriesDataset(BaseTimeSeriesDataset):
    datasets: list
    probabilities: list[float]
    context_length: int = 512
    prediction_length: int = 64
    model_type: str = "pretrain"
    mode: str = "train"
    np_dtype: np.dtype = np.dtype(np.float32)
    num_test_instances: int = 1
    window_style: str = "sampled"
    window_start: int = 0
    window_stride: int = 1
    transforms: list[Callable] | None = None
    augmentations: list[Callable] | None = None
    augmentation_rate: float = 0.0
    augmentation_probabilities: list[float] | None = None
    random_seed: int = 1337

    def __post_init__(self):
        assert len(self.probabilities) == len(self.datasets)
        assert self.mode in ("train", "test")
        self._rng = np.random.default_rng(self.random_seed)
        self.splitter = self._create_instance_splitter(self.mode)

        if self.augmentations is None:
            return

        if self.augmentation_probabilities is None:
            self.augmentation_probabilities = [1.0 / len(self.augmentations)] * len(self.augmentations)

        assert len(self.augmentations) == len(self.augmentation_probabilities)
        assert sum(self.augmentation_probabilities) == 1.0

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def preprocess_entry(self, entry: dict) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)

        if self.mode == "train" and self.augmentations is not None:
            if self.rng.random() < self.augmentation_rate:
                augmentation_idx = self.rng.choice(len(self.augmentations), p=self.augmentation_probabilities)
                entry["target"] = self.augmentations[augmentation_idx](entry["target"])

        for transform in self.transforms or []:
            entry["target"] = transform(entry["target"])

        # there's only one preprocessed entry
        for prepro_entry in self.splitter.apply([entry], is_train=self.mode == "train"):
            return prepro_entry

    def _preprocessed_datasets(self) -> list:
        return [Map(self.preprocess_entry, dataset) for dataset in self.datasets]

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"], dtype=torch.float32)

        if self.model_type == "pretrain":
            return {"past_values": past_target}

        future_target = torch.tensor(entry["future_target"], dtype=torch.float32)
        return {"past_values": past_target, "future_values": future_target}

    def _format_train(self, entry: dict) -> dict:
        return self.to_hf_format(entry)

    def _format_eval(self, entry: dict) -> dict:
        return self.to_hf_format(entry)
