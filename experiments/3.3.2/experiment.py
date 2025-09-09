from fogbed import HardwareResources, CloudResourceModel, EdgeResourceModel
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import LinkResources, calculate_compute_units, cu_with_margin
from task import MainTask


task = MainTask()
num_devices = task.train_configs().num_clients

host_cpu_ghz = 2.25

server_cpu_ghz = 2.0
server_memory_mb = 2048
server_network_mbps = 1000

pi3_cpu_ghz = 1.2
pi3_memory_mb = 1024
pi3_network_mbps = 100

server_cu = calculate_compute_units(host_cpu_ghz, server_cpu_ghz)
server_mu = server_memory_mb
server_bw = server_network_mbps

pi3_cu = calculate_compute_units(host_cpu_ghz, pi3_cpu_ghz)
pi3_mu = pi3_memory_mb
pi3_bw = pi3_network_mbps

cloud_cu = cu_with_margin(server_cu)
cloud_mu = server_mu

edge_cu = cu_with_margin(pi3_cu * num_devices)
edge_mu = pi3_mu * num_devices

exp_cu = cu_with_margin(cloud_cu + edge_cu)
exp_mu = cloud_mu + edge_mu

exp = NetflExperiment(
	name="exp-3.3.2",
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

devices = exp.create_devices(
	"pi3",
	HardwareResources(cu=pi3_cu, mu=pi3_mu),
	LinkResources(bw=pi3_bw),
	num_devices
)

exp.add_docker(server, cloud)
for device in devices: exp.add_docker(device, edge)

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
