import tensorflow as tf
from keras import models, optimizers
from flwr.server.strategy import FedAvg

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
			label_dtype=tf.int32
		)
	
	def dataset_partitioner(self) -> DatasetPartitioner:
		return PathologicalPartitioner(
			num_classes_per_partition=4,
			class_assignment_mode='deterministic'
		)

	def normalized_dataset(self, raw_dataset: Dataset) -> Dataset:
		mean = tf.constant([0.4914, 0.4822, 0.4465], shape=(1, 1, 1, 3), dtype=tf.float32)
		std = tf.constant([0.2470, 0.2435, 0.2616], shape=(1, 1, 1, 3), dtype=tf.float32)
		x = tf.cast(raw_dataset.x, tf.float32) / 255.0
		x = (x - mean) / std
		return Dataset(
			x=x,
			y=raw_dataset.y
		)

	def model(self) -> models.Model:
		return cnn3(
			input_shape=(32, 32, 3),
			output_classes=10,
			optimizer=optimizers.SGD(learning_rate=0.01)
		)

	def aggregation_strategy(self) -> type[FedAvg]:
		return FedAvg
	
	def train_configs(self) -> TrainConfigs:
		return TrainConfigs(
			batch_size=16,
			epochs=2,
			num_clients=64,
			num_partitions=64,
			num_rounds=500,
			seed_data=42,
			shuffle_data=True
		)


class MainTask(Cifar10):
	pass
