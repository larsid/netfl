from fogbed import HardwareResources
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import (
	LinkConfigs, 
	create_edge_resources,
	create_cloud_resources,
	create_experiment_resources
)
from task import MainTask


task = MainTask()

server_resources = HardwareResources(cu=1.0, mu=1024)

total_edge_0_devices = 2
device_edge_0_resources = HardwareResources(cu=0.25, mu=512)

total_edge_1_devices = 2
device_edge_1_resources = HardwareResources(cu=0.25, mu=512)

cloud_resources = create_cloud_resources([server_resources])
edge_0_resources = create_edge_resources(
    [device_edge_0_resources for _ in range(total_edge_0_devices)]
)
edge_1_resources = create_edge_resources(
    [device_edge_1_resources for _ in range(total_edge_1_devices)]
)
exp_resources = create_experiment_resources(
    [cloud_resources, edge_0_resources, edge_1_resources]
)

server_cloud_link = LinkConfigs(bw=1000)
device_edge_0_link = LinkConfigs(bw=100)
device_edge_1_link = LinkConfigs(bw=50)

cloud_edge_0_link = LinkConfigs(bw=10)
cloud_edge_1_link = LinkConfigs(bw=5)

exp = NetflExperiment("mnist-exp", task, exp_resources)

cloud = exp.add_virtual_instance("cloud", cloud_resources)
edge_0 = exp.add_virtual_instance("edge_0", edge_0_resources)
edge_1 = exp.add_virtual_instance("edge_1", edge_1_resources)

server = exp.create_server("server", server_resources, server_cloud_link)

edge_0_devices = exp.create_devices(
    "edge_0_device", device_edge_0_resources, device_edge_0_link, total_edge_0_devices
)

edge_1_devices = exp.create_devices(
    "edge_1_device", device_edge_1_resources, device_edge_1_link, total_edge_1_devices
)

exp.add_docker(server, cloud)
for device in edge_0_devices: exp.add_docker(device, edge_0)
for device in edge_1_devices: exp.add_docker(device, edge_1)

worker = exp.add_worker("127.0.0.1", port=5000)

worker.add(cloud)
worker.add(edge_0)
worker.add(edge_1)

worker.add_link(cloud, edge_0, cloud_edge_0_link)
worker.add_link(cloud, edge_1, cloud_edge_1_link)

try:
    exp.start()
except Exception as ex: 
    print(ex)
finally:
    exp.stop()
