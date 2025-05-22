from dataclasses import dataclass
from typing import Any

from fogbed import HardwareResources, CloudResourceModel, EdgeResourceModel
from fogbed.resources.protocols import ResourceModel


@dataclass
class LinkConfigs:
	bw: int | None = None
	delay: str | None = None
	loss: int | None = None

	def to_params(self) -> dict[str, Any]:
		return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class ExperimentResourceModel:
	max_cu: float
	max_mu: int


def create_edge_resources(resources: list[HardwareResources]) -> EdgeResourceModel:
	total_cu = sum(r.compute_units for r in resources)
	total_mu = sum(r.memory_units for r in resources)
	return EdgeResourceModel(max_cu=total_cu, max_mu=total_mu)


def create_cloud_resources(resources: list[HardwareResources]) -> CloudResourceModel:
	total_cu = sum(r.compute_units for r in resources)
	total_mu = sum(r.memory_units for r in resources)
	return CloudResourceModel(max_cu=total_cu, max_mu=total_mu)


def create_experiment_resources(resources: list[ResourceModel]) -> ExperimentResourceModel:
	total_cu = sum(r.max_cu for r in resources)
	total_mu = sum(r.max_mu for r in resources)
	return ExperimentResourceModel(max_cu=total_cu, max_mu=total_mu)
