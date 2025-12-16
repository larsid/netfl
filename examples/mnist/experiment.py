from netfl.core.experiment import FLExperiment
from netfl.utils.resources import (
    WorkerHostResource,
    NetworkResource,
    DeviceResource,
    ClusterResource,
    ClusterResourceType,
)

from task import FLTask


task = FLTask()
clients_per_edge = task.train_configs().num_clients // 2

worker_host_resource = WorkerHostResource()

server_resource = DeviceResource(
    name="server",
    cpu_cores=8,
    cpu_clock=2.0,
    memory=4096,
    network_resource=NetworkResource(bw=1000),
    worker_host_resource=worker_host_resource,
)

client_a_resource = DeviceResource(
    name="client_a",
    cpu_cores=4,
    cpu_clock=1.2,
    memory=1024,
    network_resource=NetworkResource(bw=100),
    worker_host_resource=worker_host_resource,
)

client_b_resource = DeviceResource(
    name="client_b",
    cpu_cores=4,
    cpu_clock=1.5,
    memory=2048,
    network_resource=NetworkResource(bw=1000),
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
    device_resources=clients_per_edge * [client_a_resource],
)

edge_1_resource = ClusterResource(
    name="edge_1",
    type=ClusterResourceType.EDGE,
    device_resources=clients_per_edge * [client_b_resource],
)

exp = FLExperiment(
    name="mnist-exp",
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

worker = exp.register_remote_worker(ip="127.0.0.1", port=5000)
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
