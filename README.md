# NetFL

**NetFL** is a framework that extends [Fogbed](https://github.com/larsid/fogbed) by integrating [Flower](https://github.com/adap/flower), enabling simulation of Federated Learning experiments within Fog/Edge computing environments. It supports the modeling of heterogeneous and resource-constrained edge scenarios, incorporating factors such as computational disparities among clients and dynamic network conditions, including bandwidth limitations, latency variations, and packet loss. This facilitates realistic evaluations of FL systems under non-ideal, real-world conditions.

## Installation

> **Requirements**: Ubuntu 22.04 LTS or later, Python 3.9.

### 1. Set up Containernet

Refer to the [Containernet documentation](https://github.com/containernet/containernet) for further details.

Install Ansible:

```
sudo apt-get install ansible
```

Clone the Containernet repository:

```
git clone https://github.com/containernet/containernet.git
```

Run the installation playbook:

```
sudo ansible-playbook -i "localhost," -c local containernet/ansible/install.yml
```

Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

> **Note:** The virtual environment **must be activated** before installing or using any Python packages, including Containernet and NetFL.

Install Containernet into the active virtual environment:

```
pip install containernet/.
```

### 2. Install NetFL

While the virtual environment is still active, run:

```
pip install netfl
```

## Running an Experiment with NetFL and Fogbed

Follow the steps below to set up and run an experiment using **NetFL**. This is an example using the **MNIST** dataset. You can find more examples in the `examples` folder:

### 1. Define the Dataset, the Model, and the Training Configurations

```py
from keras import layers, models
from flwr.server.strategy import FedAvg

from netfl.core.task import Task, Dataset, DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.utils.partitioner import IidPartitioner


class MNIST(Task):
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            huggingface_path="ylecun/mnist",
            item_name="image",
            label_name="label",
        )
    
    def dataset_partitioner(self) -> DatasetPartitioner:
        return IidPartitioner()

    def normalized_dataset(self, raw_dataset: Dataset) -> Dataset:
        return Dataset(
            x=(raw_dataset.x / 255.0),
            y=raw_dataset.y,
        )

    def model(self) -> models.Model:
        model = models.Sequential([
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        return model

    def aggregation_strategy(self) -> type[FedAvg]:
        return FedAvg
    
    def train_configs(self) -> TrainConfigs:
	    return TrainConfigs(
            batch_size=32,
            epochs=1,
            fraction_fit=1.0,
            learning_rate=0.001,
            min_available=4,
            max_available=4,
            num_rounds=10,
            seed=42,
            shuffle=True,
            test_size=0.2,
        )


class MainTask(MNIST):
    pass

```

### 2. Start Fogbed Workers and Define the Experiment Network Topology

Refer to the [Fogbed documentation](https://larsid.github.io/fogbed/distributed_emulation) for detailed instructions on starting workers.

![Network Topology](https://i.postimg.cc/3r2k2W90/network-topology.png)

### 3. Create and Run the Experiment

```py
from fogbed import CloudResourceModel, EdgeResourceModel, HardwareResources

from netfl.core.experiment import NetflExperiment
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
    print("The experiment is running...")
    input("Press enter to finish")
except Exception as ex: 
    print(ex)
finally:
    exp.stop()

```

## Running a Simple Example with a Basic Network Topology Using Docker

### 1. Create the Main Task

In the project root directory, create or modify a **NetFL Task** and name the file `task.py`. Refer to the examples in the `examples` folder for guidance on task creation.

### 2. Create the Infrastructure

Use Docker Compose to set up the infrastructure, including the server and clients:

```
docker compose up -d
```

### 3. View Training Results

To check the server logs, run:

```
docker logs server
```

Training logs are also stored in the logs folder within the project root directory. 

### 4. Shut Down the Infrastructure

To stop and remove all running containers, use the following command:

```
docker compose down
```

## More information

- [NetFL on PyPI](https://pypi.org/project/netfl)

- [NetFL Docker Images](https://hub.docker.com/r/netfl/netfl/tags)
