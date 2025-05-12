from uuid import uuid4 as uuid
from typing import Any

from fogbed import FogbedDistributedExperiment, Container
from fogbed.resources.flavors import HardwareResources, Resources

from netfl.core.task import Task
from netfl.utils.initializer import EXPERIMENT_ENV_VAR, get_task_dir


class NetflExperiment(FogbedDistributedExperiment):
	def __init__(self, 
		main_task: type[Task],
		dimage: str = "netfl/netfl",
		server_port: int = 9191,
		controller_ip: str | None = None,
    	controller_port: int = 6633,
		max_cpu: float = 1,
		max_memory: int = 512,
		metrics_enabled: bool = False,
	):
		super().__init__(controller_ip, controller_port, max_cpu, max_memory, metrics_enabled)
		
		self._experiment_id = str(uuid())
		self._task = main_task()
		self._task_dir = get_task_dir(self._task)
		self._dimage = dimage
		self._server_port = server_port
		self._server: Container | None = None
		self._devices: list[Container] = []

	@property
	def experiment_id(self) -> str:
		return self._experiment_id

	def create_server(
		self, 
		ip: str | None = None,
		port: int | None = None,
		resources: HardwareResources = Resources.SMALL,
		link_params: dict[str, Any] = {},
    	**params: Any,
	) -> Container:
		if self._server is not None:
			raise RuntimeError("The experiment already has a server.")
		
		if port is not None:
			self._server_port = port
		
		self._server = Container(
			name="server", 
			ip=ip,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=server --server_port={self._server_port}",
			environment={EXPERIMENT_ENV_VAR: self._experiment_id},
			port_bindings={self._server_port:self._server_port},
			volumes=[
				f"{self._task_dir}/task.py:/app/task.py",
				f"{self._task_dir}/logs:/app/logs"
			],
			resources=resources,
			link_params=link_params,
			params=params,
		)

		return self._server

	def create_device(
		self,
		ip: str | None = None,
		resources: HardwareResources = Resources.SMALL,
		link_params: dict[str, Any] = {},
    	**params: Any,
	) -> Container:
		if self._server is None:
			raise RuntimeError("The server must be created before creating devices.")

		if len(self._devices) + 1 > self._task._train_configs.max_available:
			raise RuntimeError(f"The maximum number of devices ({self._task._train_configs.max_available}) has been reached.")
		
		device_id = len(self._devices)
		device = Container(
			name=f"device_{device_id}",
			ip=ip,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=client --client_id={device_id} --server_address={self._server.ip} --server_port={self._server_port}",
			environment={EXPERIMENT_ENV_VAR: self._experiment_id},
			resources=resources,
			link_params=link_params,
			params=params,
		)
		self._devices.append(device)

		return device
