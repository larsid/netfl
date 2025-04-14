import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from keras import models
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import ndarrays_to_parameters, Parameters, Metrics
from flwr.server.strategy import Strategy, FedAvg


@dataclass
class TrainConfig:
	batch_size: int
	epochs: int
	fraction_evaluate: float
	fraction_fit: float
	learning_rate: float
	min_available: int
	max_available: int
	num_rounds: int
	seed: int
	shuffle: bool
	test_size: float


@dataclass
class DatasetInfo:
	huggingface_path: str
	item_name: str
	label_name: str


@dataclass
class Dataset:
	x_train: np.ndarray
	y_train: np.ndarray
	x_test: np.ndarray
	y_test: np.ndarray


class Task(ABC):
	def __init__(self) -> None:
		self._train_config = self.train_config()
		self._dataset_info = self.dataset_info()

		if self._train_config.min_available < 2:
			raise ValueError(f"train_config.min_available must be at least 2, got {self._train_config.min_available}.")
		
		if self._train_config.min_available > self._train_config.max_available:
			raise ValueError("train_config.min_available must be less than or equal to train_config.max_available.")
		
		self._fldataset = FederatedDataset(
			dataset= self._dataset_info.huggingface_path,
			partitioners={
				"train": IidPartitioner(num_partitions=self._train_config.max_available)
			},
			seed=self._train_config.seed,
			shuffle=self._train_config.shuffle,
		)
	
	def _dataset_partition(self, client_id: int) -> Dataset:
		if (client_id >= self._train_config.max_available):
			raise ValueError(f"client_id must be less than train_config.max_available, got {client_id}.")
		
		partition = self._fldataset.load_partition(partition_id=client_id)
		partition.set_format("numpy")
		partition = partition.train_test_split(
			seed=self._train_config.seed,
			shuffle=self._train_config.shuffle,
			test_size=self._train_config.test_size,
		)
		x_train, y_train = (
			np.array(partition["train"][self._dataset_info.item_name]),
			np.array(partition["train"][self._dataset_info.label_name]),
		)
		x_test, y_test = (
			np.array(partition["test"][self._dataset_info.item_name]),
			np.array(partition["test"][self._dataset_info.label_name]),
		)
		return Dataset(x_train, y_train, x_test, y_test)
	
	def _model_parameters(self) -> Parameters:
		return ndarrays_to_parameters(self.model().get_weights())
	
	def _aggregation_evaluate_metrics(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
		formatted_metrics = sorted((x[1] for x in metrics), key=lambda d: d["client_id"])
		logging.info("clients_evaluate_metrics: %s", formatted_metrics)
		return {}

	def _aggregation_strategy_factory(self, cls: type[FedAvg]) -> Strategy:
		return cls(
			evaluate_metrics_aggregation_fn=self._aggregation_evaluate_metrics,
			fit_metrics_aggregation_fn=lambda metrics: {},
			fraction_evaluate=self._train_config.fraction_evaluate,
			fraction_fit=self._train_config.fraction_fit,
			initial_parameters=self._model_parameters(),
			min_available_clients=self._train_config.min_available,
		)
	
	@abstractmethod
	def dataset_info(self) -> DatasetInfo:
		pass

	@abstractmethod
	def dataset(self, raw_dataset: Dataset) -> Dataset:
		pass

	@abstractmethod
	def model(self) -> models.Model:
		pass

	@abstractmethod
	def aggregation_strategy(self) -> Strategy:
		pass

	@abstractmethod
	def train_config(self) -> TrainConfig:
		pass
