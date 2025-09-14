from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import NetworkResource, Resource, ClusterResource, ClusterResourceType

from task import MainTask


task = MainTask()
train_configs = task.train_configs()

if train_configs.num_devices % 2 != 0: 
	raise ValueError("Expected an even number of devices.")

host_cpu_clock = 2.25

server_resource = Resource(
	name="server",
	cpus=14,
	cpu_clock=2.0,
	host_cpu_clock=host_cpu_clock,
	memory=8 * 1024,
	network=NetworkResource(bw=1000)
)

pi3_resource = Resource(
	name="pi3",
	cpus=4,
	cpu_clock=1.2,
	host_cpu_clock=host_cpu_clock,
	memory=1024,
	network=NetworkResource(bw=100)
)

pi4_resource = Resource(
	name="pi4",
	cpus=4,
	cpu_clock=1.5,
	host_cpu_clock=host_cpu_clock,
	memory=4 * 1024,
	network=NetworkResource(bw=100)
)

cloud_resource = ClusterResource(
	name="cloud",
	type=ClusterResourceType.CLOUD,
	resources=[server_resource]
)

edge_resource = ClusterResource(
	name="edge",
	type=ClusterResourceType.EDGE,
	resources=(train_configs.num_devices // 2) * [pi3_resource, pi4_resource]
)

exp = NetflExperiment(
	name="exp-3.1.3",
	task=task,
	resources=[cloud_resource, edge_resource]
)

cloud = exp.create_cluster(cloud_resource)
edge = exp.create_cluster(edge_resource)

server = exp.create_server(server_resource)
pi3_devices = exp.create_devices(pi3_resource, edge_resource.num_resources // 2)
pi4_devices = exp.create_devices(pi4_resource, edge_resource.num_resources // 2)

exp.add_to_cluster(server, cloud)
for device in pi3_devices: exp.add_to_cluster(device, edge)
for device in pi4_devices: exp.add_to_cluster(device, edge)

worker = exp.add_worker("127.0.0.1")
worker.add(cloud)
worker.add(edge)
worker.add_link(cloud, edge)

try:
	exp.start()
except Exception as ex: 
	print(ex)
finally:
	exp.stop()
