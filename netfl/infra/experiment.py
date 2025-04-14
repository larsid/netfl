from fogbed import FogbedDistributedExperiment, Container
from fogbed.resources.flavors import HardwareResources, Resources

from netfl.core.task import Task
from netfl.errors.exceptions import ServerAlreadyExistsError, ServerNotCreatedError, MaxDevicesReachedError
from netfl.utils.initializer import get_task_dir


class Experiment(FogbedDistributedExperiment):
	def __init__(self, 
		main_task: Task,
		dimage: str = "netfl/netfl",
		server_port: int = 9191,
		controller_ip: str | None = None,
    	controller_port: int = 6633,
		max_cpu: float = 1,
		max_memory: int = 512,
		metrics_enabled: bool = False,
	):
		super().__init__(controller_ip, controller_port, max_cpu, max_memory, metrics_enabled)

		self._main_task_dir = get_task_dir(main_task)
		self._main_task = main_task
		self._dimage = dimage
		self._server_port = server_port
		self._server: Container | None = None
		self._devices: list[Container] = []

	def create_server(
		self, 
		ip: str | None = None,
		port: int | None = None,
		resources: HardwareResources = Resources.SMALL,
	) -> Container:
		if self._server is not None:
			raise ServerAlreadyExistsError()
		
		if port is not None:
			self._server_port = port
		
		self._server = Container(
			name="server", 
			ip=ip,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=server --server_port={self._server_port}",
			port_bindings={self._server_port:self._server_port},
			volumes=[
				f"{self._main_task_dir}/task.py:/app/task.py",
				f"{self._main_task_dir}/logs:/app/logs"
			],
			resources=resources,
		)

		return self._server

	def create_device(
		self,
		ip: str | None = None,
		resources: HardwareResources = Resources.SMALL,
	) -> Container:
		if self._server is None:
			raise ServerNotCreatedError()

		if len(self._devices) + 1 > self._main_task._train_config.max_available:
			raise MaxDevicesReachedError(self._main_task._train_config.max_available)
		
		device_id = len(self._devices)
		device = Container(
			name=f"device_{device_id}",
			ip=ip,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=client --client_id={device_id} --server_address={self._server.ip} --server_port={self._server_port}",
			resources=resources,
		)
		self._devices.append(device)

		return device
