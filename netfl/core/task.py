import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any

import tensorflow as tf
from keras import models
from datasets import DownloadConfig
from flwr_datasets import FederatedDataset, partitioner
from flwr.server.strategy import Strategy

from netfl.utils.log import log
from netfl.utils.net import execute


@dataclass
class TrainConfigs:
	batch_size: int
	epochs: int
	num_devices: int
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

		if self._train_configs.num_devices > self._train_configs.num_partitions:
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
		strategy_type, strategy_args = self.aggregation_strategy()
		strategy_configs = {**strategy_args, "name": strategy_type.__name__}

		log(f"[DATASET INFO]\n{json.dumps(asdict(self._dataset_info), indent=2, default=str)}")
		log(f"[DATASET PARTITIONER CONFIGS]\n{json.dumps(self._dataset_partitioner_configs, indent=2, default=str)}")
		log(f"[AGGREGATION STRATEGY CONFIGS]\n{json.dumps(strategy_configs, indent=2, default=str)}")
		log(f"[TRAIN CONFIGS]\n{json.dumps(asdict(self._train_configs), indent=2, default=str)}")

	def train_dataset(self, client_id: int) -> Dataset:
		if (client_id >= self._train_configs.num_partitions):
			raise ValueError(f"The client_id must be less than num_partitions, got {client_id}.")
		
		partition = execute(lambda: self._fldataset.load_partition(client_id, "train").with_format("numpy"))

		input_key = self._dataset_info.input_key
		label_key = self._dataset_info.label_key

		input_dtype = self._dataset_info.input_dtype
		label_dtype = self._dataset_info.label_dtype

		x = tf.convert_to_tensor(partition[input_key], dtype=input_dtype)
		y = tf.convert_to_tensor(partition[label_key], dtype=label_dtype)

		return self.preprocess_dataset(Dataset(x, y), True)

	def test_dataset(self) -> Dataset:
		test_dataset =  execute(lambda: self._fldataset.load_split("test").with_format("numpy"))

		input_key = self._dataset_info.input_key
		label_key = self._dataset_info.label_key

		input_dtype = self._dataset_info.input_dtype
		label_dtype = self._dataset_info.label_dtype

		x = tf.convert_to_tensor(test_dataset[input_key], dtype=input_dtype)
		y = tf.convert_to_tensor(test_dataset[label_key], dtype=label_dtype)

		return self.preprocess_dataset(Dataset(x, y), False)

	def batch_dataset(self, dataset: Dataset) -> tuple[tf.data.Dataset, int]:
		length = int(dataset.x.shape[0])  # type: ignore[index]

		batch_dataset = (
			tf.data.Dataset.from_tensor_slices((dataset.x, dataset.y))
			.shuffle(buffer_size=length)
			.batch(self._train_configs.batch_size)
			.prefetch(tf.data.AUTOTUNE)
		)

		return (batch_dataset, length)

	@abstractmethod
	def dataset_info(self) -> DatasetInfo:
		"""Provides metadata about the dataset to be used.

		This method should return a `DatasetInfo` object that specifies all the
		necessary details for loading the dataset, such as its path on the
		Hugging Face Hub, the names of the input and label columns, and the
		expected data types.

		Returns:
			DatasetInfo: An object containing the dataset's configuration.
		"""
		pass

	@abstractmethod
	def dataset_partitioner(self) -> DatasetPartitioner:
		"""Defines the partitioning strategy for distributing the dataset.

		This method should return an instance of a `DatasetPartitioner` subclass
		that defines how the training data will be split among the virtual
		clients (e.g., IID, Pathological Non-IID, etc.).

		Returns:
			DatasetPartitioner: An object that will create the data splits.
		"""
		pass

	@abstractmethod
	def preprocess_dataset(self, dataset: Dataset, training: bool) -> Dataset:
		"""Applies preprocessing steps to the dataset.

		This method defines a pipeline for data transformations, such
		as normalization and data augmentation. The `training` flag
		allows for the conditional application of operations that should
		only be performed on the training set (e.g., augmentation).

		Args:
			dataset (Dataset): The input `Dataset` object, expected to have `x` (features)
				and `y` (labels) attributes.
			training (bool): A flag indicating if training-specific preprocessing
				should be applied. Set to `True` for the training set.

		Returns:
			Dataset: A new `Dataset` object with the transformed features.
		"""
		pass

	@abstractmethod
	def model(self) -> models.Model:
		"""Defines and compiles the machine learning model architecture.

		This method should construct, compile, and return a `keras.Model`
		instance. The definition includes the layers of the neural network
		as well as the optimizer, loss function, and metrics.

		Returns:
			keras.Model: The compiled Keras model to be trained.
		"""
		pass

	@abstractmethod
	def aggregation_strategy(self) -> tuple[type[Strategy], dict[str, Any]]:
		"""Specifies the federated learning aggregation strategy and its arguments.

		This method defines the core federated algorithm to be used by the server
		for aggregating client updates.

		Returns:
			Tuple[Type[Strategy], Dict[str, Any]]: A tuple where the first element
				is the strategy class (e.g., `FedAvg`, `FedProx`) and the second
				is a dictionary of its keyword arguments (e.g., `{"proximal_mu": 1.0}`).
		"""
		pass

	@abstractmethod
	def train_configs(self) -> TrainConfigs:
		"""Provides the hyperparameters for the federated training process.

		This method should return a `TrainConfigs` object that contains all
		the necessary settings for the simulation, such as the number of rounds,
		clients, local epochs, and batch size.

		Returns:
			TrainConfigs: An object containing the federated training hyperparameters.
		"""
		pass
