from fogbed import FogbedDistributedExperiment, Container, HardwareResources
from fogbed.emulation import Services

from netfl.core.task import Task
from netfl.utils.initializer import EXPERIMENT_ENV_VAR, get_task_dir
from netfl.utils.resources import LinkResources


class NetflExperiment(FogbedDistributedExperiment):
	def __init__(
		self,
		name: str,
		task: Task,
		max_cu: float,
		max_mu: int,
		dimage: str = "netfl/netfl",
		controller_ip: str | None = None,
		controller_port: int = 6633,
		metrics_enabled: bool = False,
	):
		super().__init__(
			controller_ip=controller_ip,
			controller_port=controller_port,
			max_cpu=max_cu,
			max_memory=max_mu,
			metrics_enabled=metrics_enabled
		)
		
		self._name = name
		self._task = task
		self._task_dir = get_task_dir(self._task)
		self._dimage = dimage
		self._server: Container | None = None
		self._server_port: int | None = None
		self._devices: list[Container] = []

	@property
	def name(self) -> str:
		return self._name

	def create_server(
		self,
		name: str,
		resources: HardwareResources,
		link: LinkResources,
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
			link_params=link.params,
		)
		self._server_port = port

		return self._server

	def create_device(
		self,
		name: str,
		resources: HardwareResources,
		link: LinkResources,
	) -> Container:
		if self._server is None:
			raise RuntimeError("The server must be created before creating devices.")

		if len(self._devices) + 1 > self._task._train_configs.num_clients:
			raise RuntimeError(f"The number of devices ({self._task._train_configs.num_clients}) has been reached.")
		
		device_id = len(self._devices)
		device = Container(
			name=name,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=client --client_id={device_id} --client_name={name} --server_address={self._server.ip} --server_port={self._server_port}",
			environment={EXPERIMENT_ENV_VAR: self._name},
			resources=resources,
			link_params=link.params,
			params={"--memory-swap": resources.memory_units * 2},
		)
		self._devices.append(device)

		return device

	def create_devices(
		self,
		name: str,
		resources: HardwareResources,
		link: LinkResources,
		total: int,
	) -> list[Container]:
		if total <= 0:
			raise RuntimeError(f"The total devices ({total}) must be greater than zero.")

		return [
			self.create_device(name=f"{name}_{i}", resources=resources, link=link)
			for i in range(total)
		]

	def start(self) -> None:
		print(f"Experiment is running")
		print(f"Experiment {self._name}: (cu={Services.get_all_compute_units()}, mu={Services.get_all_memory_units()})")

		for instance in self.get_virtual_instances():
			print(f"\tInstance {instance.label}: (cu={instance.compute_units}, mu={instance.memory_units})")
			for container in instance.containers.values():
				print(
					f"\t\tContainer {container.name}: "
					f"(cu={container.compute_units}, mu={container.memory_units}), "
					f"(cq={container.cpu_quota}, cp={container.cpu_period})"
				)

		super().start()
		input("Press enter to stop the experiment...")
