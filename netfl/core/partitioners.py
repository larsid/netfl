from typing import Any, Literal

from flwr_datasets import partitioner

from netfl.core.task import DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.external import partitioners


class IidPartitioner(DatasetPartitioner):
	def partitioner(
		self,
		dataset_info: DatasetInfo,
		train_configs: TrainConfigs,
	) -> tuple[dict[str, Any], partitioner.Partitioner]:
		configs = {
			"name": self.__class__.__name__,
			"num_partitions": train_configs.num_partitions,
		}

		return configs, partitioner.IidPartitioner(
			num_partitions=train_configs.num_partitions,
		)


class DirichletPartitioner(DatasetPartitioner):
	def __init__(
			self,
			alpha: float,
			min_partition_size: int = 0,
			self_balancing: bool = True,
		):
		self.alpha = alpha
		self.min_partition_size = min_partition_size
		self.self_balancing = self_balancing

	def partitioner(
		self,
		dataset_info: DatasetInfo,
		train_configs: TrainConfigs,
	) -> tuple[dict[str, Any], partitioner.Partitioner]:
		configs = {
			"name": self.__class__.__name__,
			"alpha": self.alpha,
			"min_partition_size": self.min_partition_size,
			"self_balancing": self.self_balancing,
			"partition_by": dataset_info.label_key,
			"num_partitions": train_configs.num_partitions,
			"seed_data": train_configs.seed_data,
			"shuffle_data": train_configs.shuffle_data,
		}

		return configs, partitioner.DirichletPartitioner(
			alpha=self.alpha,
			min_partition_size=self.min_partition_size,
			self_balancing=self.self_balancing,
			partition_by=dataset_info.label_key,
			num_partitions=train_configs.num_partitions,
			seed=train_configs.seed_data,
			shuffle=train_configs.shuffle_data,
		)


class PathologicalPartitioner(DatasetPartitioner):
	def __init__(
		self,
		num_classes_per_partition: int,
		class_assignment_mode: Literal[
			"random", "deterministic", "first-deterministic"
		],
	):
		self.num_classes_per_partition = num_classes_per_partition
		self.class_assignment_mode = class_assignment_mode

	def partitioner(
		self,
		dataset_info: DatasetInfo,
		train_configs: TrainConfigs,
	) -> tuple[dict[str, Any], partitioner.Partitioner]:
		configs = {
			"name": self.__class__.__name__,
			"num_classes_per_partition": self.num_classes_per_partition,
			"class_assignment_mode": self.class_assignment_mode,
			"partition_by": dataset_info.label_key,
			"num_partitions": train_configs.num_partitions,
			"seed_data": train_configs.seed_data,
			"shuffle": train_configs.shuffle_data,
		}
		
		return configs, partitioners.PathologicalPartitioner(
			num_classes_per_partition=self.num_classes_per_partition,
			class_assignment_mode=self.class_assignment_mode, # type: ignore[arg-type]
			partition_by=dataset_info.label_key,
			num_partitions=train_configs.num_partitions,
			seed=train_configs.seed_data,
			shuffle=train_configs.shuffle_data,
		)
