import tensorflow as tf
from keras import models, optimizers
from flwr.server.strategy import FedAvg

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
			label_dtype=tf.int32
		)
	
	def dataset_partitioner(self) -> DatasetPartitioner:
		return IidPartitioner()

	def normalized_dataset(self, raw_dataset: Dataset) -> Dataset:
		return Dataset(
			x=tf.cast(raw_dataset.x, tf.float32) / 255.0,
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
			num_clients=4,
			num_partitions=4,
			num_rounds=10,
			seed_data=42,
			shuffle_data=True
		)


class MainTask(Cifar10):
	pass
