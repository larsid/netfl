from fogbed import HardwareResources, CloudResourceModel, EdgeResourceModel
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import LinkResources, calculate_compute_units, max_compute_unit
from task import MainTask


task = MainTask()
num_devices = task.train_configs().num_clients
num_devices_type_one = num_devices // 2
num_devices_type_two = num_devices // 2

host_cpu_ghz = 2.25

server_cpu_ghz = 2.0
server_memory_mb = 2048
server_network_mbps = 1000

device_type_one_cpu_ghz = 1.2
device_type_one_memory_mb = 1024
device_type_one_network_mbps = 100

device_type_two_cpu_ghz = 1.5
device_type_two_memory_mb = 2048
device_type_two_network_mbps = 1000

server_cu = calculate_compute_units(host_cpu_ghz, server_cpu_ghz)
server_mu = server_memory_mb
server_bw = server_network_mbps

device_type_one_cu = calculate_compute_units(host_cpu_ghz, device_type_one_cpu_ghz)
device_type_one_mu = device_type_one_memory_mb
device_type_one_bw = device_type_one_network_mbps

device_type_two_cu = calculate_compute_units(host_cpu_ghz, device_type_two_cpu_ghz)
device_type_two_mu = device_type_two_memory_mb
device_type_two_bw = device_type_two_network_mbps

cloud_cu = max_compute_unit(server_cu)
cloud_mu = server_mu

edge_cu = max_compute_unit((device_type_one_cu * num_devices_type_one) + (device_type_two_cu * num_devices_type_two))
edge_mu = (device_type_one_mu * num_devices_type_one) + (device_type_two_mu * num_devices_type_two)

exp_cu = max_compute_unit(cloud_cu + edge_cu)
exp_mu = cloud_mu + edge_mu

exp = NetflExperiment(
	name="exp-2.1.1",
	task=task,
	max_cu=exp_cu,
	max_mu=exp_mu
)

cloud = exp.add_virtual_instance(
	"cloud",
	CloudResourceModel(max_cu=cloud_cu, max_mu=cloud_mu)
)

edge = exp.add_virtual_instance(
	"edge",
	EdgeResourceModel(max_cu=edge_cu, max_mu=edge_mu)
)

server = exp.create_server(
	"server",
	HardwareResources(cu=server_cu, mu=server_mu),
	LinkResources(bw=server_bw),
)

devices_type_one = exp.create_devices(
	"device1",
	HardwareResources(cu=device_type_one_cu, mu=device_type_one_mu),
	LinkResources(bw=device_type_one_bw),
	num_devices_type_one
)

devices_type_two = exp.create_devices(
	"device2",
	HardwareResources(cu=device_type_two_cu, mu=device_type_two_mu),
	LinkResources(bw=device_type_two_bw),
	num_devices_type_two
)

exp.add_docker(server, cloud)
for device in devices_type_one: exp.add_docker(device, edge)
for device in devices_type_two: exp.add_docker(device, edge)

worker = exp.add_worker("127.0.0.1", port=5000)
worker.add(cloud)
worker.add(edge)
worker.add_link(cloud, edge)

try:
	exp.start()
except Exception as ex: 
	print(ex)
finally:
	exp.stop()
