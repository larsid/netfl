from dataclasses import dataclass
from typing import Any
from enum import Enum

from fogbed.resources.protocols import ResourceModel
from fogbed import CloudResourceModel, FogResourceModel, EdgeResourceModel


COMPUTE_UNIT_PRECISION = 6
COMPUTE_UNIT_ERROR = 1 / 10 ** (COMPUTE_UNIT_PRECISION + 1)


@dataclass
class WorkerHostResource:
    cpu_cores: int = 1
    cpu_clock: float = 1.0


@dataclass
class NetworkResource:
    bw: int | None = None
    delay: str | None = None
    loss: int | None = None

    @property
    def link_params(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if v is not None}


@dataclass
class DeviceResource:
    name: str
    cpu_cores: int
    cpu_clock: float
    memory: int
    network_resource: NetworkResource
    worker_host_resource: WorkerHostResource

    @property
    def compute_units(self) -> float:
        device_ccr = self.cpu_cores * self.cpu_clock
        host_ccr = (
            self.worker_host_resource.cpu_cores * self.worker_host_resource.cpu_clock
        )

        if device_ccr <= 0 or host_ccr <= 0:
            raise ValueError("Clock-Cycle Rates must be greater than zero.")

        return round(device_ccr / host_ccr, COMPUTE_UNIT_PRECISION)

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
    device_resources: list[DeviceResource]

    @property
    def num_devices(self) -> int:
        return len(self.device_resources)

    @property
    def resource_model(self) -> ResourceModel:
        max_cu = (
            sum(r.compute_units for r in self.device_resources) + COMPUTE_UNIT_ERROR
        )
        max_mu = sum(r.memory_units for r in self.device_resources)

        match self.type:
            case ClusterResourceType.CLOUD:
                return CloudResourceModel(max_cu, max_mu)
            case ClusterResourceType.FOG:
                return FogResourceModel(max_cu, max_mu)
            case ClusterResourceType.EDGE:
                return EdgeResourceModel(max_cu, max_mu)
