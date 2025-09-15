from dataclasses import replace

from fogbed import FogbedDistributedExperiment, VirtualInstance, Container, HardwareResources
from fogbed.emulation import Services

from netfl.core.task import Task
from netfl.utils.initializer import EXPERIMENT_ENV_VAR, get_task_dir
from netfl.utils.resources import Resource, ClusterResource


class NetflExperiment(FogbedDistributedExperiment):
	def __init__(
		self,
		name: str,
		task: Task,
		resources: list[ClusterResource],
		dimage: str = "netfl/netfl",
		controller_ip: str | None = None,
		controller_port: int = 6633,
		metrics_enabled: bool = False,
	):
		resource_models = [r.resource_model for r in resources]
		max_cu = sum(r.max_cu for r in resource_models)
		max_mu = sum(r.max_mu for r in resource_models)

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

	def create_cluster(self, resource: ClusterResource) -> VirtualInstance:
		virtual_instance = self.add_virtual_instance(
			name=resource.name,
			resource_model=resource.resource_model,
		)

		return virtual_instance

	def create_server(
		self,
		resource: Resource,
		ip: str | None = None,
		port: int = 9191,
	) -> Container:
		if self._server is not None:
			raise RuntimeError("The experiment already has a server.")
		
		self._server = Container(
			name=resource.name,
			ip=ip,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=server --server_port={port}",
			environment={EXPERIMENT_ENV_VAR: self._name},
			port_bindings={port:port},
			volumes=[
				f"{self._task_dir}/task.py:/app/task.py",
				f"{self._task_dir}/logs:/app/logs"
			],
			resources=HardwareResources(cu=resource.compute_units, mu=resource.memory_units),
			link_params=resource.network.link_params,
		)
		self._server_port = port

		return self._server

	def create_device(
		self,
		resource: Resource,
	) -> Container:
		if self._server is None:
			raise RuntimeError("The server must be created before creating devices.")

		if len(self._devices) + 1 > self._task._train_configs.num_devices:
			raise RuntimeError(f"The number of devices ({self._task._train_configs.num_devices}) has been reached.")
		
		device_id = len(self._devices)
		device = Container(
			name=resource.name,
			dimage=self._dimage,
			dcmd=f"python -u run.py --type=client --client_id={device_id} --client_name={resource.name} --server_address={self._server.ip} --server_port={self._server_port}",
			environment={EXPERIMENT_ENV_VAR: self._name},
			resources=HardwareResources(cu=resource.compute_units, mu=resource.memory_units),
			link_params=resource.network.link_params,
			params={"--memory-swap": resource.memory_units * 2},
		)
		self._devices.append(device)

		return device

	def create_devices(
		self,
		resource: Resource,
		total: int,
	) -> list[Container]:
		if total <= 0:
			raise RuntimeError(f"The total devices ({total}) must be greater than zero.")

		return [
			self.create_device(resource=replace(resource, name=f"{resource.name}_{i}"))
			for i in range(total)
		]

	def add_to_cluster(self, container: Container, virtual_instance: VirtualInstance) -> None: 
		self.add_docker(container=container, datacenter=virtual_instance)

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
