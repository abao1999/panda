"""
Dataset for PatchTST
"""

import itertools
from dataclasses import dataclass
from functools import partial
from typing import Iterator, List, Optional

import numpy as np
import torch
from gluonts.itertools import Cyclic, Map
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    MissingValueImputation,
    TestSplitSampler,
    ValidationSplitSampler,
)
from torch.utils.data import IterableDataset, get_worker_info


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
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


@dataclass
class PatchTSTDataset(IterableDataset):
    datasets: list
    probabilities: List[float]
    context_length: int = 512
    prediction_length: int = 64
    min_past: Optional[int] = None
    imputation_method: Optional[MissingValueImputation] = None
    mode: str = "train"
    np_dtype: np.dtype = np.dtype(np.float32)
    fixed_dim: Optional[int] = None
    delay_embed_prob: float = 0.0

    def __post_init__(self):
        assert len(self.probabilities) == len(self.datasets)
        assert self.mode in ("train", "validation", "test")

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)

        if mode == "train" and isinstance(self.fixed_dim, int):
            total_dims = entry["target"].shape[0]
            sampled_dims = np.random.choice(
                total_dims, size=self.fixed_dim, replace=False
            )
            entry["target"] = entry["target"][sampled_dims]

        if mode == "train" and self.delay_embed_prob > np.random.rand():
            # Randomly select a dimension
            num_dims = entry["target"].shape[0]
            selected_dim = np.random.randint(num_dims)
            selected_series = entry["target"][selected_dim]

            # Create delay embeddings
            delay_embeddings = np.stack(
                [np.roll(selected_series, shift) for shift in range(num_dims)]
            )[:, num_dims - 1 :]  # cut off rolling artifacts

            # Replace the original target with delay embeddings
            entry["target"] = delay_embeddings

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["train", "test", "validation"]

        instance_sampler = {
            "train": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.context_length,  # never sample behind the timeseries
                min_future=self.prediction_length,  # never sample too far ahead
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
        split_transform = self._create_instance_splitter("train")
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"], dtype=torch.float32)
        future_target = torch.tensor(entry["future_target"], dtype=torch.float32)

        return {
            "past_values": past_target,
            "future_values": future_target,
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "train":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
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
                    data = next(iterators[idx])
                    yield self.to_hf_format(data)
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)
