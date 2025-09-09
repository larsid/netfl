# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file is a **modified version** of the `PathologicalPartitioner` class
# originally implemented by Flower Labs:
# https://github.com/adap/flower/blob/main/datasets/flwr_datasets/partitioner/pathological_partitioner.py
#
# Modifications in this version include:
# - Enforcing equal-sized sample splits per class across assigned partitions
# - Truncating remainder samples to achieve uniformity
# - Clean separation of class assignment modes
# - Added safety checks to prevent assignment overflows
#
# These changes aim to improve experimental determinism and partition balance
# for simulation-based federated learning setups.


import warnings
from typing import Any, Literal, Optional

import numpy as np
import datasets
from flwr_datasets.common.typing import NDArray
from flwr_datasets.partitioner.partitioner import Partitioner


class PathologicalPartitioner(Partitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        num_classes_per_partition: int,
        class_assignment_mode: Literal[
            "random", "deterministic", "first-deterministic"
        ] = "random",
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._num_classes_per_partition = num_classes_per_partition
        self._class_assignment_mode = class_assignment_mode
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_unique_labels: dict[int, list[Any]] = {
            pid: [] for pid in range(self._num_partitions)
        }
        self._unique_labels: list[Any] = []
        self._unique_label_to_times_used_counter: dict[Any, int] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        if self._partition_id_to_indices_determined:
            return
        self._determine_partition_id_to_unique_labels()
        self._count_partitions_having_each_unique_label()
        labels = np.asarray(self.dataset[self._partition_by])
        self._check_correctness_of_unique_label_to_times_used_counter(labels)
        for partition_id in range(self._num_partitions):
            self._partition_id_to_indices[partition_id] = []
        unused_labels = []
        for unique_label in self._unique_labels:
            if self._unique_label_to_times_used_counter[unique_label] == 0:
                unused_labels.append(unique_label)
                continue
            unique_label_to_indices = np.where(labels == unique_label)[0]
            if self._shuffle:
                self._rng.shuffle(unique_label_to_indices)
            num_splits = self._unique_label_to_times_used_counter[unique_label]
            samples_per_split = len(unique_label_to_indices) // num_splits
            trimmed = unique_label_to_indices[:samples_per_split * num_splits]
            split_unique_labels_to_indices = np.split(trimmed, num_splits)
            split_index = 0
            for partition_id in range(self._num_partitions):
                if unique_label in self._partition_id_to_unique_labels[partition_id]:
                    assert split_index < len(split_unique_labels_to_indices), (
                        f"Split index {split_index} out of range for label {unique_label}"
                    )
                    self._partition_id_to_indices[partition_id].extend(
                        split_unique_labels_to_indices[split_index].tolist()
                    )
                    split_index += 1
        if len(unused_labels) >= 1:
            warnings.warn(
                f"Classes: {unused_labels} will NOT be used due to the chosen "
                f"configuration. If it is undesired behavior consider setting"
                f" 'first_class_deterministic_assignment=True' which in case when"
                f" the number of classes is smaller than the number of partitions will "
                f"utilize all the classes for the created partitions.",
                stacklevel=1,
            )
        for partition_id, indices in self._partition_id_to_indices.items():
            if self._shuffle:
                self._rng.shuffle(indices)
        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _determine_partition_id_to_unique_labels(self) -> None:
        self._unique_labels = sorted(self.dataset.unique(self._partition_by))
        num_classes = len(self._unique_labels)
        total_class_assignments = self._num_partitions * self._num_classes_per_partition
        base_usage = total_class_assignments // num_classes
        remainder = total_class_assignments % num_classes
        class_usage_counter = {
            label: base_usage + (1 if i < remainder else 0)
            for i, label in enumerate(self._unique_labels)
        }
        repeated_labels = []
        for label, count in class_usage_counter.items():
            repeated_labels.extend([label] * count)
        if self._class_assignment_mode == "random":
            self._rng.shuffle(repeated_labels)
        elif self._class_assignment_mode == "first-deterministic":
            repeated_labels.sort()
        elif self._class_assignment_mode == "deterministic":
            label_cycle = np.tile(self._unique_labels, (total_class_assignments // num_classes) + 1)
            repeated_labels = label_cycle[:total_class_assignments].tolist()
        for partition_id in range(self._num_partitions):
            start = partition_id * self._num_classes_per_partition
            end = start + self._num_classes_per_partition
            self._partition_id_to_unique_labels[partition_id] = repeated_labels[start:end]

    def _count_partitions_having_each_unique_label(self) -> None:
        for unique_label in self._unique_labels:
            self._unique_label_to_times_used_counter[unique_label] = 0
        for unique_labels in self._partition_id_to_unique_labels.values():
            for unique_label in unique_labels:
                self._unique_label_to_times_used_counter[unique_label] += 1

    def _check_correctness_of_unique_label_to_times_used_counter(
        self, labels: NDArray
    ) -> None:
        for unique_label in self._unique_labels:
            num_unique = np.sum(labels == unique_label)
            if self._unique_label_to_times_used_counter[unique_label] > num_unique:
                raise ValueError(
                    f"Label: {unique_label} is needed to be assigned to more "
                    f"partitions "
                    f"({self._unique_label_to_times_used_counter[unique_label]})"
                    f" than there are samples (corresponding to this label) in the "
                    f"dataset ({num_unique}). Please decrease the `num_partitions`, "
                    f"`num_classes_per_partition` to avoid this situation, "
                    f"or try `class_assignment_mode='deterministic'` to create a more "
                    f"even distribution of classes along the partitions. "
                    f"Alternatively use a different dataset if you can not adjust"
                    f" the any of these parameters."
                )
