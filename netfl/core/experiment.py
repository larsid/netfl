from fogbed import Worker, VirtualInstance, Controller, FogbedDistributedExperiment, Container, HardwareResources
from fogbed.exceptions import WorkerAlreadyExists

from netfl.core.task import Task
from netfl.utils.initializer import EXPERIMENT_ENV_VAR, get_task_dir
from netfl.utils.resources import LinkConfigs, ExperimentResourceModel


class NetflWorker(Worker):
	def add_link(
		self, 
		node1: VirtualInstance, 
		node2: VirtualInstance, 
		link: LinkConfigs
	):
		super().add_link(node1, node2, **link.to_params())


class NetflExperiment(FogbedDistributedExperiment):
	def __init__(
		self,
		name: str,
		task: Task,
		resources: ExperimentResourceModel,
		dimage: str = "netfl/netfl",
		controller_ip: str | None = None,
		controller_port: int = 6633,
		metrics_enabled: bool = False,
	):
		super().__init__(
			controller_ip=controller_ip, 
			controller_port=controller_port, 
			max_cpu=resources.max_cu,
			max_memory=resources.max_mu,
			metrics_enabled=metrics_enabled
		)
		
		self._name = name
		self._task = task
		self._resources = resources
		self._task_dir = get_task_dir(self._task)
		self._dimage = dimage
		self._server: Container | None = None
		self._server_port: int | None = None
		self._devices: list[Container] = []

	@property
	def name(self) -> str:
		return self._name

	def add_worker(
		self, 
		ip: str, 
		port: int = 5000, 
		controller: Controller | None = None
	) -> NetflWorker:
		if(ip in self.workers):
			raise WorkerAlreadyExists(ip)

		worker = NetflWorker(ip, port, controller)
		self.workers[worker.ip] = worker
		return worker

	def create_server(
		self, 
		name: str,
		resources: HardwareResources,
		link: LinkConfigs,
		ip: str | None = None,
		port: int = 9191,
	) -> Container:
		if self._server is not None:
			raise RuntimeError("The experiment already has a server.")
		
		self._server = Container(
			name=name,
			ip=ip,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=server --server_port={port}",
			environment={EXPERIMENT_ENV_VAR: self._name},
			port_bindings={port:port},
			volumes=[
				f"{self._task_dir}/task.py:/app/task.py",
				f"{self._task_dir}/logs:/app/logs"
			],
			resources=resources,
			link_params=link.to_params(),
		)
		self._server_port = port

		return self._server

	def create_device(
		self,
		name: str,
		resources: HardwareResources,
		link: LinkConfigs,
	) -> Container:
		if self._server is None:
			raise RuntimeError("The server must be created before creating devices.")

		if len(self._devices) + 1 > self._task._train_configs.max_clients:
			raise RuntimeError(f"The maximum number of devices ({self._task._train_configs.max_clients}) has been reached.")
		
		device_id = len(self._devices)
		device = Container(
			name=name,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=client --client_id={device_id} --server_address={self._server.ip} --server_port={self._server_port}",
			environment={EXPERIMENT_ENV_VAR: self._name},
			resources=resources,
			link_params=link.to_params(),
		)
		self._devices.append(device)

		return device

	def start(self):
		print(f"Experiment {self._name} is running")
		print(f"Allocated resources: (compute_units={self._resources.max_cu}, memory_units={self._resources.max_mu})")
		return super().start()
