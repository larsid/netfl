from fogbed import HardwareResources, CloudResourceModel, EdgeResourceModel
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import LinkResources
from task import MainTask


exp = NetflExperiment(name="cifar10-exp", task=MainTask(), max_cu=2.0, max_mu=3072)

cloud_resources = CloudResourceModel(max_cu=1.0, max_mu=1024)
edge_0_resources = EdgeResourceModel(max_cu=0.5, max_mu=1024)
edge_1_resources = EdgeResourceModel(max_cu=0.5, max_mu=1024)

server_resources = HardwareResources(cu=1.0, mu=1024)
server_link = LinkResources(bw=1000)

edge_0_total_devices = 2
edge_0_device_resources = HardwareResources(cu=0.25, mu=512)
edge_0_device_link = LinkResources(bw=100)

edge_1_total_devices = 2
edge_1_device_resources = HardwareResources(cu=0.25, mu=512)
edge_1_device_link = LinkResources(bw=50)

cloud_edge_0_link = LinkResources(bw=10)
cloud_edge_1_link = LinkResources(bw=5)

cloud = exp.add_virtual_instance("cloud", cloud_resources)
edge_0 = exp.add_virtual_instance("edge_0", edge_0_resources)
edge_1 = exp.add_virtual_instance("edge_1", edge_1_resources)

server = exp.create_server("server", server_resources, server_link)

edge_0_devices = exp.create_devices(
	"edge_0_device", edge_0_device_resources, edge_0_device_link, edge_0_total_devices
)

edge_1_devices = exp.create_devices(
	"edge_1_device", edge_1_device_resources, edge_1_device_link, edge_1_total_devices
)

exp.add_docker(server, cloud)
for device in edge_0_devices: exp.add_docker(device, edge_0)
for device in edge_1_devices: exp.add_docker(device, edge_1)

worker = exp.add_worker("127.0.0.1", port=5000)

worker.add(cloud)
worker.add(edge_0)
worker.add(edge_1)

worker.add_link(cloud, edge_0, **cloud_edge_0_link.params)
worker.add_link(cloud, edge_1, **cloud_edge_1_link.params)

try:
	exp.start()
except Exception as ex: 
	print(ex)
finally:
	exp.stop()
