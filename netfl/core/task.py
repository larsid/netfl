import gc
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from keras import models
from datasets import DownloadConfig
from flwr_datasets import FederatedDataset, partitioner
from flwr.server.strategy import FedAvg

from netfl.utils.log import log


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
	item_name: str
	label_name: str


@dataclass
class Dataset:
	x: np.ndarray
	y: np.ndarray


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
			raise ValueError("num_clients must be less than or equal to num_partitions.")
		
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
				num_proc=1,
				max_retries=10
			),
		)

	def print_configs(self):
		log(f"[DATASET INFO]\n{json.dumps(asdict(self._dataset_info), indent=2)}")
		log(f"[DATASET PARTITIONER CONFIGS]\n{json.dumps(self._dataset_partitioner_configs, indent=2)}")
		log(f"[TRAIN CONFIGS]\n{json.dumps(asdict(self._train_configs), indent=2)}")

	def train_dataset(self, client_id: int) -> Dataset:
		if (client_id >= self._train_configs.num_partitions):
			raise ValueError(f"client_id must be less than num_partitions, got {client_id}.")
		
		partition = self._fldataset.load_partition(client_id, "train").with_format("numpy")

		x = np.array(partition[self._dataset_info.item_name])
		y = np.array(partition[self._dataset_info.label_name])

		return self.normalized_dataset(Dataset(x, y))

	def test_dataset(self) -> Dataset:
		test_dataset = self._fldataset.load_split("test").with_format("numpy")

		x = np.array(test_dataset[self._dataset_info.item_name])
		y = np.array(test_dataset[self._dataset_info.label_name])

		return self.normalized_dataset(Dataset(x, y))
	
	def delete_downloaded_dataset(self) -> None:
		del self._fldataset
		gc.collect()

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
