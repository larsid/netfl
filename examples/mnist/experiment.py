from netfl.infra.experiment import Experiment, HardwareResources
from task import MainTask

exp = Experiment(main_task=MainTask())

worker = exp.add_worker(ip="worker-ip", port=5000)

cloud  = exp.add_virtual_instance("cloud")
edge_0 = exp.add_virtual_instance("edge_0")
edge_1 = exp.add_virtual_instance("edge_1")

server = exp.create_server(
    resources=HardwareResources(cu=1.0,  mu=1024),
    link_params={"bw": 1000, "delay": "1ms"},
)

edge_0_devices = [ 
    exp.create_device(
        resources=HardwareResources(cu=0.5,  mu=512),
        link_params={"bw": 100, "delay": "5ms"},
    ) for _ in range(2)
]

edge_1_devices = [ 
    exp.create_device(
        resources=HardwareResources(cu=0.5,  mu=512),
        link_params={"bw": 50, "delay": "5ms"},
    ) for _ in range(2)
]

exp.add_docker(server, cloud)

exp.add_docker(edge_0_devices[0], edge_0)
exp.add_docker(edge_0_devices[1], edge_0)

exp.add_docker(edge_1_devices[0], edge_1)
exp.add_docker(edge_1_devices[1], edge_1)

worker.add(cloud)
worker.add(edge_0)
worker.add(edge_1)

worker.add_link(
    cloud, 
    edge_0, 
    bw=10, delay="50ms", loss=1, max_queue_size=100, use_htb=True,
)

worker.add_link(
    cloud, 
    edge_1, 
    bw=5, delay="25ms", loss=1, max_queue_size=100, use_htb=True,
)

try:
    exp.start()    
    print("The experiment is running...")
    input("Press enter to finish")
except Exception as ex: 
    print(ex)
finally:
    exp.stop()
