from typing import Any

import tensorflow as tf
from keras import models, optimizers
from flwr.server.strategy import Strategy, FedAvg

from netfl.core.task import Task, Dataset, DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.core.models import cnn3
from netfl.core.partitioners import IidPartitioner


class Cifar10(Task):
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            huggingface_path="uoft-cs/cifar10",
            input_key="img",
            label_key="label",
            input_dtype=tf.float32,
            label_dtype=tf.int32,
        )

    def dataset_partitioner(self) -> DatasetPartitioner:
        return IidPartitioner()

    def preprocess_dataset(self, dataset: Dataset, training: bool) -> Dataset:
        return Dataset(x=tf.divide(dataset.x, 255.0), y=dataset.y)

    def model(self) -> models.Model:
        optimizer = optimizers.SGD(learning_rate=0.01)
        model = cnn3(input_shape=(32, 32, 3), output_classes=10, optimizer=optimizer)
        model.summary()
        print(optimizer.get_config())
        return model

    def aggregation_strategy(self) -> tuple[type[Strategy], dict[str, Any]]:
        return FedAvg, {}

    def train_configs(self) -> TrainConfigs:
        return TrainConfigs(
            batch_size=16,
            epochs=2,
            num_clients=16,
            num_partitions=64,
            num_rounds=500,
            seed_data=42,
            shuffle_data=True,
        )


class FLTask(Cifar10):
    pass
