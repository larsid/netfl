from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import (
    WorkerHostResource,
    NetworkResource,
    DeviceResource,
    ClusterResource,
    ClusterResourceType,
    BASE_COMPUTE_UNIT,
)

from task import FLTask


task = FLTask()
num_clients = task.train_configs().num_clients

worker_host_resource = WorkerHostResource(cpu_clock=BASE_COMPUTE_UNIT)

server_resource = DeviceResource(
    name="server",
    cpu_cores=1,
    cpu_clock=1.0,
    memory=1024,
    network_resource=NetworkResource(bw=1000),
    worker_host_resource=worker_host_resource,
)

client_a_resource = DeviceResource(
    name="client_a",
    cpu_cores=1,
    cpu_clock=0.5,
    memory=512,
    network_resource=NetworkResource(bw=100),
    worker_host_resource=worker_host_resource,
)

client_b_resource = DeviceResource(
    name="client_b",
    cpu_cores=1,
    cpu_clock=0.25,
    memory=512,
    network_resource=NetworkResource(bw=50),
    worker_host_resource=worker_host_resource,
)

cloud_resource = ClusterResource(
    name="cloud",
    type=ClusterResourceType.CLOUD,
    device_resources=[server_resource],
)

edge_0_resource = ClusterResource(
    name="edge_0",
    type=ClusterResourceType.EDGE,
    device_resources=(num_clients // 2) * [client_a_resource],
)

edge_1_resource = ClusterResource(
    name="edge_1",
    type=ClusterResourceType.EDGE,
    device_resources=(num_clients // 2) * [client_b_resource],
)

exp = NetflExperiment(
    name="cifar10-exp",
    task=task,
    cluster_resources=[cloud_resource, edge_0_resource, edge_1_resource],
)

server = exp.create_server(server_resource)
edge_0_clients = exp.create_clients(client_a_resource, edge_0_resource.num_devices)
edge_1_clients = exp.create_clients(client_b_resource, edge_1_resource.num_devices)

cloud = exp.create_cluster(cloud_resource)
edge_0 = exp.create_cluster(edge_0_resource)
edge_1 = exp.create_cluster(edge_1_resource)

exp.add_to_cluster(server, cloud)

for client in edge_0_clients:
    exp.add_to_cluster(client, edge_0)
for client in edge_1_clients:
    exp.add_to_cluster(client, edge_1)

worker = exp.register_remote_worker("127.0.0.1")
worker.add_cluster(cloud)
worker.add_cluster(edge_0)
worker.add_cluster(edge_1)
worker.create_cluster_link(cloud, edge_0, NetworkResource(bw=10))
worker.create_cluster_link(cloud, edge_1, NetworkResource(bw=20))

try:
    exp.start()
except Exception as ex:
    print(ex)
finally:
    exp.stop()
