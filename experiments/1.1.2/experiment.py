from fogbed import HardwareResources, CloudResourceModel, EdgeResourceModel
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import LinkResources, calculate_compute_units
from task import MainTask


task = MainTask()
num_devices = task.train_configs().num_clients

host_cpu_ghz = 3.5

server_cpu_ghz = 2
server_memory_mb = 2048
server_network_mbps = 1000

device_cpu_ghz = 1.2
device_memory_mb = 1024
device_network_mbps = 100

server_cu = calculate_compute_units(host_cpu_ghz, server_cpu_ghz)
server_mu = server_memory_mb
server_bw = server_network_mbps

device_cu = calculate_compute_units(host_cpu_ghz, device_cpu_ghz)
device_mu = device_memory_mb
device_bw = device_network_mbps

cloud_cu = server_cu
cloud_mu = server_mu

edge_cu = device_cu * num_devices
edge_mu = device_mu * num_devices

exp = NetflExperiment(
    name="exp-1.1.2", 
    task=task, 
    max_cu=cloud_cu + edge_cu, 
    max_mu=cloud_mu + edge_mu
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
    "device", 
    HardwareResources(cu=device_cu, mu=device_mu), 
    LinkResources(bw=device_bw), 
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
    input("Press enter to finish")
except Exception as ex: 
    print(ex)
finally:
    exp.stop()
