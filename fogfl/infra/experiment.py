import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import importlib

from fogbed import FogbedDistributedExperiment, Container
from fogbed.resources.flavors import HardwareResources, Resources

from fogfl.core.task import Task
from fogfl.errors.exceptions import ServerAlreadyExistsError, ServerNotCreatedError, MaxDevicesReachedError


class Experiment(FogbedDistributedExperiment):
	def __init__(self, 
		task: Task,
		dimage: str,
		server_port: int = 9191,
		controller_ip: str | None = None,
    	controller_port: int = 6633,
		max_cpu: float = 1,
		max_memory: int = 512,
		metrics_enabled: bool = False,
	):
		super().__init__(controller_ip, controller_port, max_cpu, max_memory, metrics_enabled)

		self._task = task
		self._dimage = dimage
		self._server_port = server_port
		self._server: Container | None = None
		self._devices: list[Container] = []

		self._task_dir = self._get_task_dir()
		self._validate_task_dir()

	def _get_task_dir(self) -> str:
		task_cls = self._task.__class__
		module_name = task_cls.__module__
		module = importlib.import_module(module_name)

		if hasattr(module, '__file__') and isinstance(module.__file__, str):
			return os.path.dirname(os.path.abspath(module.__file__))

		raise FileNotFoundError("Could not determine the task directory.")

	def _validate_task_dir(self) -> None:
		if not os.path.isdir(self._task_dir):
			raise FileNotFoundError(f"Task directory '{self._task_dir}' does not exist.")

		task_file = os.path.join(self._task_dir, "task.py")
		if not os.path.isfile(task_file):
			raise FileNotFoundError(f"'task.py' not found in the task directory '{self._task_dir}'.")

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
				f"{self._task_dir}/task.py:/app/task.py",
				f"{self._task_dir}/logs:/app/logs"
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

		if len(self._devices) + 1 > self._task.train_config.max_available:
			raise MaxDevicesReachedError(self._task.train_config.max_available)
		
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
