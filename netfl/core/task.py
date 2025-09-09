import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any

import tensorflow as tf
from keras import models
from datasets import DownloadConfig
from flwr_datasets import FederatedDataset, partitioner
from flwr.server.strategy import FedAvg

from netfl.utils.log import log
from netfl.utils.net import execute


@dataclass
class TrainConfigs:
	batch_size: int
	epochs: int
	num_clients: int
	num_partitions: int
	num_rounds: int
	seed_data: int
	shuffle_data: bool


@dataclass
class DatasetInfo:
	huggingface_path: str
	input_key: str
	label_key: str
	input_dtype: tf.DType
	label_dtype: tf.DType


@dataclass
class Dataset:
	x: tf.Tensor
	y: tf.Tensor


class DatasetPartitioner(ABC):
	@abstractmethod
	def partitioner(
		self,
		dataset_info: DatasetInfo,
		train_configs: TrainConfigs,
	) -> tuple[dict[str, Any], partitioner.Partitioner]:
		pass


class Task(ABC):
	def __init__(self):
		self._train_configs = self.train_configs()
		self._dataset_info = self.dataset_info()

		if self._train_configs.num_clients > self._train_configs.num_partitions:
			raise ValueError("The num_clients must be less than or equal to num_partitions.")
		
		self._dataset_partitioner_configs, self._dataset_partitioner = self.dataset_partitioner().partitioner(
			self._dataset_info,
			self._train_configs,
		)
		
		self._fldataset = FederatedDataset(
			dataset= self._dataset_info.huggingface_path,
			partitioners={
				"train": self._dataset_partitioner
			},
			seed=self._train_configs.seed_data,
			shuffle=self._train_configs.shuffle_data,
			trust_remote_code=True,
			streaming=False,
			download_config=DownloadConfig(
				max_retries=0,
				num_proc=1
			),
		)

	def print_configs(self):
		log(f"[DATASET INFO]\n{json.dumps(asdict(self._dataset_info), indent=2, default=str)}")
		log(f"[DATASET PARTITIONER CONFIGS]\n{json.dumps(self._dataset_partitioner_configs, indent=2, default=str)}")
		log(f"[TRAIN CONFIGS]\n{json.dumps(asdict(self._train_configs), indent=2, default=str)}")

	def train_dataset(self, client_id: int) -> Dataset:
		if (client_id >= self._train_configs.num_partitions):
			raise ValueError(f"The client_id must be less than num_partitions, got {client_id}.")
		
		partition = execute(lambda: self._fldataset.load_partition(client_id, "train").with_format("numpy"))

		input_key = self._dataset_info.input_key
		label_key = self._dataset_info.label_key

		input_dtype = self._dataset_info.input_dtype
		label_dtype = self._dataset_info.label_dtype

		x = tf.convert_to_tensor([sample[input_key] for sample in partition], dtype=input_dtype) # type: ignore[index]
		y = tf.convert_to_tensor([sample[label_key] for sample in partition], dtype=label_dtype)  # type: ignore[index]

		return self.normalized_dataset(Dataset(x, y))

	def test_dataset(self) -> Dataset:
		test_dataset =  execute(lambda: self._fldataset.load_split("test").with_format("numpy"))

		input_key = self._dataset_info.input_key
		label_key = self._dataset_info.label_key

		input_dtype = self._dataset_info.input_dtype
		label_dtype = self._dataset_info.label_dtype

		x = tf.convert_to_tensor([sample[input_key] for sample in test_dataset], dtype=input_dtype) # type: ignore[index]
		y = tf.convert_to_tensor([sample[label_key] for sample in test_dataset], dtype=label_dtype)  # type: ignore[index]

		return self.normalized_dataset(Dataset(x, y))

	def batch_dataset(self, dataset: Dataset) -> tuple[tf.data.Dataset, int]:
		length = int(dataset.x.shape[0])  # type: ignore[index]

		batch_dataset = (
			tf.data.Dataset.from_tensor_slices((dataset.x, dataset.y))
			.batch(self._train_configs.batch_size)
			.prefetch(tf.data.AUTOTUNE)
		)

		return (batch_dataset, length)

	@abstractmethod
	def dataset_info(self) -> DatasetInfo:
		pass

	@abstractmethod
	def dataset_partitioner(self) -> DatasetPartitioner:
		pass

	@abstractmethod
	def normalized_dataset(self, raw_dataset: Dataset) -> Dataset:
		pass

	@abstractmethod
	def model(self) -> models.Model:
		pass

	@abstractmethod
	def aggregation_strategy(self) -> type[FedAvg]:
		pass

	@abstractmethod
	def train_configs(self) -> TrainConfigs:
		pass
