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
    verbose: str


@dataclass
class DatasetConfig:
    dataset_name: str
    item_name: str
    label_name: str


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class Task(ABC):
    @abstractmethod
    def __init__(
        self,
        train_config: TrainConfig,
        dataset_config: DatasetConfig,
    ) -> None:
        if train_config.min_available < 2:
            raise ValueError(f"train_config.min_available must be at least 2, got {train_config.min_available}")

        self._train_config = train_config
        self._dataset_config = dataset_config
        self._dataset = FederatedDataset(
            dataset= self._dataset_config.dataset_name,
            partitioners={
                "train": IidPartitioner(num_partitions=self._train_config.max_available)
            },
            seed=self._train_config.seed,
            shuffle=self._train_config.shuffle,
        )

    @property
    def train_config(self) -> TrainConfig:
        return self._train_config
    
    @property
    def dataset_config(self) -> DatasetConfig:
        return self._dataset_config
    
    def _dataset_partition(self, client_id: int) -> Dataset:
        if (client_id >= self._train_config.max_available):
            raise ValueError(f"client_id must be less than train_config.max_available, got {client_id}")
        partition = self._dataset.load_partition(partition_id=client_id)
        partition.set_format("numpy")
        partition = partition.train_test_split(
            seed=self._train_config.seed,
            shuffle=self._train_config.shuffle,
            test_size=self._train_config.test_size,
        )
        x_train, y_train = (
            np.array(partition["train"][self._dataset_config.item_name]),
            np.array(partition["train"][self._dataset_config.label_name]),
        )
        x_test, y_test = (
            np.array(partition["test"][self._dataset_config.item_name]),
            np.array(partition["test"][self._dataset_config.label_name]),
        )
        return Dataset(x_train, y_train, x_test, y_test)
    
    def _model_parameters(self) -> Parameters:
        return ndarrays_to_parameters(self.model().get_weights())
    
    def _aggregation_strategy_factory(self, cls: type[FedAvg]) -> Strategy:
        return cls(
            evaluate_metrics_aggregation_fn=self.aggregation_evaluate_metrics,
            fit_metrics_aggregation_fn=lambda metrics: {},
            fraction_evaluate=self._train_config.fraction_evaluate,
            fraction_fit=self._train_config.fraction_fit,
            initial_parameters=self._model_parameters(),
            min_available_clients=self._train_config.min_available,
        )

    @abstractmethod
    def client_dataset(self, client_id: int) -> Dataset:
        pass

    @abstractmethod
    def model(self) -> models.Model:
        pass

    @abstractmethod
    def aggregation_strategy(self) -> Strategy:
        pass

    @abstractmethod
    def aggregation_evaluate_metrics(self, metrics: list[tuple[int, Metrics]]) -> Metrics:
        pass
