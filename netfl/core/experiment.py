from typing import Any
from dataclasses import replace

from fogbed import (
    FogbedDistributedExperiment,
    VirtualInstance,
    Container,
    HardwareResources,
)
from fogbed.emulation import Services
from fogbed.node.controller import Controller
from fogbed.exceptions import WorkerAlreadyExists

from netfl.core.task import Task
from netfl.core.worker import Worker
from netfl.utils.initializer import EXPERIMENT_ENV_VAR, get_task_dir
from netfl.utils.resources import DeviceResource, ClusterResource


class NetflExperiment(FogbedDistributedExperiment):
    def __init__(
        self,
        name: str,
        task: Task,
        cluster_resources: list[ClusterResource],
        dimage: str = "netfl/netfl",
        hugging_face_token: str | None = None,
        controller_ip: str | None = None,
        controller_port: int = 6633,
    ):
        resource_models = [r.resource_model for r in cluster_resources]
        max_cu = sum(r.max_cu for r in resource_models)
        max_mu = sum(r.max_mu for r in resource_models)

        super().__init__(
            controller_ip=controller_ip,
            controller_port=controller_port,
            max_cpu=max_cu,
            max_memory=max_mu,
            metrics_enabled=False,
        )

        self._name = name
        self._task = task
        self._task_dir = get_task_dir(self._task)
        self._dimage = dimage
        self._hugging_face_token = hugging_face_token
        self._server: Container | None = None
        self._server_port: int | None = None
        self._clients: list[Container] = []

    @property
    def name(self) -> str:
        return self._name

    def _environment(self) -> dict[str, Any]:
        environment = {EXPERIMENT_ENV_VAR: self._name}

        if self._hugging_face_token is not None:
            environment["HF_TOKEN"] = self._hugging_face_token

        return environment

    def create_cluster(self, resource: ClusterResource) -> VirtualInstance:
        virtual_instance = self.add_virtual_instance(
            name=resource.name,
            resource_model=resource.resource_model,
        )

        return virtual_instance

    def create_server(
        self,
        resource: DeviceResource,
        ip: str | None = None,
        port: int = 9191,
    ) -> Container:
        if self._server is not None:
            raise RuntimeError("The experiment already has a server.")

        self._server = Container(
            name=resource.name,
            ip=ip,
            dimage=self._dimage,
            dcmd=(f"python -u run.py " f"--type=server " f"--server_port={port}"),
            environment=self._environment(),
            port_bindings={port: port},
            volumes=[
                f"{self._task_dir}/task.py:/app/task.py",
                f"{self._task_dir}/logs:/app/logs",
            ],
            resources=HardwareResources(
                cu=resource.compute_units, mu=resource.memory_units
            ),
            link_params=resource.network_resource.link_params,
        )
        self._server_port = port

        return self._server

    def create_client(
        self,
        resource: DeviceResource,
    ) -> Container:
        if self._server is None:
            raise RuntimeError("The server must be created before creating clients.")

        if len(self._clients) + 1 > self._task._train_configs.num_clients:
            raise RuntimeError(
                f"The number of clients ({self._task._train_configs.num_clients}) has been reached."
            )

        client_id = len(self._clients)
        client = Container(
            name=resource.name,
            dimage=self._dimage,
            dcmd=(
                f"python -u run.py "
                f"--type=client "
                f"--client_id={client_id} "
                f"--client_name={resource.name} "
                f"--server_address={self._server.ip} "
                f"--server_port={self._server_port} "
            ),
            environment=self._environment(),
            resources=HardwareResources(
                cu=resource.compute_units, mu=resource.memory_units
            ),
            link_params=resource.network_resource.link_params,
            params={"--memory-swap": resource.memory_units * 2},
        )
        self._clients.append(client)

        return client

    def create_clients(
        self,
        resource: DeviceResource,
        total: int,
    ) -> list[Container]:
        if total <= 0:
            raise RuntimeError(
                f"The total clients ({total}) must be greater than zero."
            )

        return [
            self.create_client(resource=replace(resource, name=f"{resource.name}_{i}"))
            for i in range(total)
        ]

    def add_to_cluster(
        self, container: Container, virtual_instance: VirtualInstance
    ) -> None:
        self.add_docker(container=container, datacenter=virtual_instance)

    def register_remote_worker(
        self, ip: str, port: int = 5000, controller: Controller | None = None
    ) -> Worker:
        if ip in self.workers:
            raise WorkerAlreadyExists(ip)

        worker = Worker(ip, port, controller)
        self.workers[worker.ip] = worker
        return worker

    def start(self) -> None:
        print(f"Experiment {self._name} is running")
        print(
            f"Experiment: (cu={Services.get_all_compute_units()}, mu={Services.get_all_memory_units()})"
        )

        for instance in self.get_virtual_instances():
            print(
                f"\tCluster {instance.label}: (cu={instance.compute_units}, mu={instance.memory_units})"
            )
            for container in instance.containers.values():
                print(
                    f"\t\tDevice {container.name}: "
                    f"(cu={container.compute_units}, mu={container.memory_units}), "
                    f"(cq={container.cpu_quota}, cp={container.cpu_period})"
                )

        super().start()
        input("Press enter to stop the experiment...")
