from netfl.infra.experiment import Experiment
from task import MainTask

exp = Experiment(
	main_task=MainTask()
)

worker_0 = exp.add_worker(ip="worker-ip", port=5000)

cloud  = exp.add_virtual_instance("cloud")
edge_0 = exp.add_virtual_instance("edge_0")
edge_1 = exp.add_virtual_instance("edge_1")

server = exp.create_server()

devices = [exp.create_device() for _ in range(4)]

exp.add_docker(server, cloud)

exp.add_docker(devices[0], edge_0)
exp.add_docker(devices[1], edge_0)

exp.add_docker(devices[2], edge_1)
exp.add_docker(devices[3], edge_1)

worker_0.add(cloud)
worker_0.add(edge_0)
worker_0.add(edge_1)

worker_0.add_link(cloud, edge_0, delay="10ms")
worker_0.add_link(cloud, edge_1, delay="20ms")

try:
    exp.start()    
    print("The experiment is running...")
    input("Press enter to finish")
except Exception as ex: 
    print(ex)
finally:
    exp.stop()
