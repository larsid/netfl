from dataclasses import dataclass
from typing import Any
from enum import Enum

from fogbed.resources.protocols import ResourceModel
from fogbed import CloudResourceModel, FogResourceModel, EdgeResourceModel


COMPUTE_UNIT_BASE_CLOCK = 1.0

COMPUTE_UNIT_PRECISION = 3
COMPUTE_UNIT_ERROR = 1 / 10 ** (COMPUTE_UNIT_PRECISION + 1)


def clock_to_compute_units(device_cpu_clock: float, host_cpu_clock: float) -> float:
	if host_cpu_clock <= 0 or device_cpu_clock <= 0:
		raise ValueError("CPU clocks must be greater than zero.")
	if device_cpu_clock > host_cpu_clock:
		raise ValueError(f"Device CPU clock cannot exceed host clock.")

	return round(device_cpu_clock / host_cpu_clock, COMPUTE_UNIT_PRECISION)


@dataclass
class Host:
	cpu_clock: float = COMPUTE_UNIT_BASE_CLOCK


@dataclass
class NetworkResource:
	bw: int | None = None
	delay: str | None = None
	loss: int | None = None

	@property
	def link_params(self) -> dict[str, Any]:
		return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class Resource:
	name: str
	cpu_cores: int
	cpu_clock: float
	memory: int
	network: NetworkResource
	host: Host

	@property
	def compute_units(self) -> float:
		return clock_to_compute_units(self.cpu_clock, self.host.cpu_clock) * self.cpu_cores
	
	@property
	def memory_units(self) -> int:
		return self.memory


class ClusterResourceType(str, Enum):
	CLOUD = "cloud"
	FOG = "fog"
	EDGE = "edge"


@dataclass
class ClusterResource:
	name: str
	type: ClusterResourceType
	resources: list[Resource]

	@property
	def num_resources(self) -> int:
		return len(self.resources)

	@property
	def resource_model(self) -> ResourceModel:
		max_cu = sum(r.compute_units for r in self.resources) + COMPUTE_UNIT_ERROR
		max_mu = sum(r.memory_units for r in self.resources)

		match self.type:
			case ClusterResourceType.CLOUD:
				return CloudResourceModel(max_cu, max_mu)
			case ClusterResourceType.FOG:
				return FogResourceModel(max_cu, max_mu)
			case ClusterResourceType.EDGE:
				return EdgeResourceModel(max_cu, max_mu)
