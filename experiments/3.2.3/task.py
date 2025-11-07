from typing import Any

import tensorflow as tf
from keras import models, optimizers
from flwr.server.strategy import Strategy, FedAvg

from netfl.core.task import Task, Dataset, DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.core.models import cnn3
from netfl.core.partitioners import PathologicalPartitioner


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
        return PathologicalPartitioner(
            num_classes_per_partition=4, class_assignment_mode="deterministic"
        )

    def preprocess_dataset(self, dataset: Dataset, training: bool) -> Dataset:
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        mean = tf.reshape(mean, (1, 1, 3))
        std = tf.reshape(std, (1, 1, 3))

        x = tf.divide(dataset.x, 255.0)
        x = tf.subtract(x, mean)
        x_normalized = tf.divide(x, std)

        return Dataset(x=x_normalized, y=dataset.y)

    def model(self) -> models.Model:
        return cnn3(
            input_shape=(32, 32, 3),
            output_classes=10,
            optimizer=optimizers.SGD(learning_rate=0.01),
        )

    def aggregation_strategy(self) -> tuple[type[Strategy], dict[str, Any]]:
        return FedAvg, {}

    def train_configs(self) -> TrainConfigs:
        return TrainConfigs(
            batch_size=16,
            epochs=2,
            num_clients=32,
            num_partitions=64,
            num_rounds=500,
            seed_data=42,
            shuffle_data=True,
        )


class FLTask(Cifar10):
    pass
