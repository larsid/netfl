from fogbed import CloudResourceModel, EdgeResourceModel, HardwareResources
from netfl.infra.experiment import Experiment
from task import MainTask

exp = Experiment(
    main_task=MainTask(),
    max_cpu=2.0,
    max_memory=2048,
)

worker = exp.add_worker(ip="192.168.0.100", port=5000)

cloud = exp.add_virtual_instance(
    name="cloud",
    resource_model=CloudResourceModel(max_cu=1.0, max_mu=1024)
)

edge_0 = exp.add_virtual_instance(
    name="edge_0",
    resource_model=EdgeResourceModel(max_cu=0.5, max_mu=512)
)

edge_1 = exp.add_virtual_instance(
    name="edge_1",
    resource_model=EdgeResourceModel(max_cu=0.5, max_mu=512)
)

server = exp.create_server(
    resources=HardwareResources(cu=0.5,  mu=512),
    link_params={"bw": 1000, "delay": "1ms"},
)

edge_0_devices = [ 
    exp.create_device(
        resources=HardwareResources(cu=0.25,  mu=256),
        link_params={"bw": 100, "delay": "5ms"},
    ) for _ in range(2)
]

edge_1_devices = [ 
    exp.create_device(
        resources=HardwareResources(cu=0.25,  mu=256),
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
