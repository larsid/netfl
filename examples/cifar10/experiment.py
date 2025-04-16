from netfl.infra.experiment import Experiment, HardwareResources
from task import MainTask

exp = Experiment(
	main_task=MainTask()
)

worker_0 = exp.add_worker(ip="worker-ip", port=5000)

cloud  = exp.add_virtual_instance("cloud")
edge_0 = exp.add_virtual_instance("edge_0")
edge_1 = exp.add_virtual_instance("edge_1")

server = exp.create_server(resources=HardwareResources(cu=1.0,  mu=512))

devices = [
    exp.create_device(resources=HardwareResources(cu=0.5,  mu=128)) 
    for _ in range(4)
]

exp.add_docker(server, cloud)

exp.add_docker(devices[0], edge_0)
exp.add_docker(devices[1], edge_0)

exp.add_docker(devices[2], edge_1)
exp.add_docker(devices[3], edge_1)

worker_0.add(cloud)
worker_0.add(edge_0)
worker_0.add(edge_1)

worker_0.add_link(
    cloud, 
    edge_0, 
    bw=1, delay="5ms", loss=1, max_queue_size=100, use_htb=True,
)

worker_0.add_link(
    cloud, 
    edge_1, 
    bw=2, delay="5ms", loss=1, max_queue_size=100, use_htb=True,
)

try:
    exp.start()    
    print("The experiment is running...")
    input("Press enter to finish")
except Exception as ex: 
    print(ex)
finally:
    exp.stop()
