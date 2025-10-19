import os

from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import (
    WorkerHostResource,
    NetworkResource,
    DeviceResource,
    ClusterResource,
    ClusterResourceType,
)

from task import FLTask


task = FLTask()
num_clients = task.train_configs().num_clients

if num_clients % 2 != 0:
    raise ValueError("Expected an even number of clients.")

worker_host_resource = WorkerHostResource(cpu_clock=2.25)

server_resource = DeviceResource(
    name="server",
    cpu_cores=14,
    cpu_clock=2.0,
    memory=2048,
    network_resource=NetworkResource(bw=1000),
    worker_host_resource=worker_host_resource,
)

pi3_resource = DeviceResource(
    name="pi3",
    cpu_cores=4,
    cpu_clock=1.2,
    memory=1024,
    network_resource=NetworkResource(bw=100),
    worker_host_resource=worker_host_resource,
)

pi4_resource = DeviceResource(
    name="pi4",
    cpu_cores=4,
    cpu_clock=1.5,
    memory=4096,
    network_resource=NetworkResource(bw=100),
    worker_host_resource=worker_host_resource,
)

cloud_resource = ClusterResource(
    name="cloud",
    type=ClusterResourceType.CLOUD,
    device_resources=[server_resource],
)

edge_resource = ClusterResource(
    name="edge",
    type=ClusterResourceType.EDGE,
    device_resources=(num_clients // 2) * [pi3_resource, pi4_resource],
)

exp = NetflExperiment(
    name="exp-3.1.1",
    task=task,
    cluster_resources=[cloud_resource, edge_resource],
    hugging_face_token=os.getenv("HUGGINGFACE_TOKEN"),
)

server = exp.create_server(server_resource)
pi3_clients = exp.create_clients(pi3_resource, edge_resource.num_devices // 2)
pi4_clients = exp.create_clients(pi4_resource, edge_resource.num_devices // 2)

cloud = exp.create_cluster(cloud_resource)
edge = exp.create_cluster(edge_resource)

exp.add_to_cluster(server, cloud)

for client in pi3_clients:
    exp.add_to_cluster(client, edge)
for client in pi4_clients:
    exp.add_to_cluster(client, edge)

worker = exp.register_remote_worker("127.0.0.1")
worker.add_cluster(cloud)
worker.add_cluster(edge)
worker.create_cluster_link(cloud, edge)

try:
    exp.start()
except Exception as ex:
    print(ex)
finally:
    exp.stop()
