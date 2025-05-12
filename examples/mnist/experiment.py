from fogbed import CloudResourceModel, EdgeResourceModel, HardwareResources

from netfl import NetflExperiment
from task import MainTask


exp = NetflExperiment(
    main_task=MainTask,
    max_cpu=2.0,
    max_memory=3072,
)

worker = exp.add_worker(ip="127.0.0.1", port=5000)

cloud = exp.add_virtual_instance(
    name="cloud",
    resource_model=CloudResourceModel(max_cu=1.0, max_mu=1024)
)

edge_0 = exp.add_virtual_instance(
    name="edge_0",
    resource_model=EdgeResourceModel(max_cu=0.5, max_mu=1024)
)

edge_1 = exp.add_virtual_instance(
    name="edge_1",
    resource_model=EdgeResourceModel(max_cu=0.5, max_mu=1024)
)

server = exp.create_server(
    resources=HardwareResources(cu=1.0,  mu=1024),
    link_params={"bw": 1000, "delay": "2ms"},
)

edge_0_devices = [ 
    exp.create_device(
        resources=HardwareResources(cu=0.25,  mu=512),
        link_params={"bw": 100, "delay": "10ms"},
    ) for _ in range(2)
]

edge_1_devices = [ 
    exp.create_device(
        resources=HardwareResources(cu=0.25,  mu=512),
        link_params={"bw": 50, "delay": "5ms"},
    ) for _ in range(2)
]

exp.add_docker(server, cloud)

for device in edge_0_devices:
    exp.add_docker(device, edge_0)

for device in edge_1_devices:
    exp.add_docker(device, edge_1)

worker.add(cloud)
worker.add(edge_0)
worker.add(edge_1)

worker.add_link(
    cloud, 
    edge_0, 
    bw=10, delay="100ms", loss=1, max_queue_size=100, use_htb=True,
)

worker.add_link(
    cloud, 
    edge_1, 
    bw=5, delay="50ms", loss=1, max_queue_size=100, use_htb=True,
)

try:
    exp.start()    
    print(f"The experiment {exp.experiment_id} is running...")
    input("Press enter to finish")
except Exception as ex: 
    print(ex)
finally:
    exp.stop()
