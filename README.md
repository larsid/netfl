# NetFL

**NetFL** is a framework for executing *Federated Learning* (FL) experiments in **simulated IoT and Fog/Edge computing environments**.
It enables the modeling of **heterogeneous and resource-constrained scenarios**, incorporating factors such as computational disparities among devices, limited bandwidth, latency, packet loss, and diverse network topologies.

Through its **native abstractions for tasks, devices, and networks**, NetFL allows researchers to configure and execute FL experiments in a **declarative and reproducible** manner, providing more realistic evaluations of algorithms under non-ideal, real-world conditions.

Under the hood, NetFL leverages [Fogbed](https://github.com/larsid/fogbed) for distributed network emulation and [Flower](https://github.com/adap/flower) for federated learning orchestration. These libraries provide robust foundations for virtualization and FL training, and NetFL integrates and extends them into a **unified framework designed specifically for FL research in IoT and Edge Computing**.


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
```

```
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

Follow the steps below to set up and run an experiment using **NetFL**. This is an example using the **MNIST** dataset. You can find more examples in the [examples](./examples/) folder:

### 1. Define the Dataset, Model, and Training Configurations

```py
from typing import Any

import tensorflow as tf
from keras import models, optimizers
from flwr.server.strategy import Strategy, FedAvg

from netfl.core.task import Task, Dataset, DatasetInfo, DatasetPartitioner, TrainConfigs
from netfl.core.models import cnn3
from netfl.core.partitioners import IidPartitioner


class MNIST(Task):
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            huggingface_path="ylecun/mnist",
            input_key="image",
            label_key="label",
            input_dtype=tf.float32,
            label_dtype=tf.int32,
        )

    def dataset_partitioner(self) -> DatasetPartitioner:
        return IidPartitioner()

    def preprocess_dataset(self, dataset: Dataset, training: bool) -> Dataset:
        return Dataset(x=tf.divide(dataset.x, 255.0), y=dataset.y)

    def model(self) -> models.Model:
        return cnn3(
            input_shape=(28, 28, 1),
            output_classes=10,
            optimizer=optimizers.SGD(learning_rate=0.01),
        )

    def aggregation_strategy(self) -> tuple[type[Strategy], dict[str, Any]]:
        return FedAvg, {}

    def train_configs(self) -> TrainConfigs:
        return TrainConfigs(
            batch_size=16,
            epochs=2,
            num_clients=4,
            num_partitions=4,
            num_rounds=10,
            seed_data=42,
            shuffle_data=True,
        )


class FLTask(MNIST):
    pass

```

### 2. Define the Experiment

![Network Topology](https://i.postimg.cc/3r2k2W90/network-topology.png)


```py
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import (
    WorkerHostResource,
    NetworkResource,
    DeviceResource,
    ClusterResource,
    ClusterResourceType,
    BASE_COMPUTE_UNIT,
)

from task import FLTask


task = FLTask()
num_clients = task.train_configs().num_clients

worker_host_resource = WorkerHostResource(cpu_clock=BASE_COMPUTE_UNIT)

server_resource = DeviceResource(
    name="server",
    cpu_cores=1,
    cpu_clock=1.0,
    memory=1024,
    network_resource=NetworkResource(bw=1000),
    worker_host_resource=worker_host_resource,
)

client_0_resource = DeviceResource(
    name="client_0",
    cpu_cores=1,
    cpu_clock=0.25,
    memory=512,
    network_resource=NetworkResource(bw=100),
    worker_host_resource=worker_host_resource,
)

client_1_resource = DeviceResource(
    name="client_1",
    cpu_cores=1,
    cpu_clock=0.25,
    memory=512,
    network_resource=NetworkResource(bw=50),
    worker_host_resource=worker_host_resource,
)

cloud_resource = ClusterResource(
    name="cloud", 
    type=ClusterResourceType.CLOUD, 
    devices=[server_resource],
)

edge_0_resource = ClusterResource(
    name="edge_0",
    type=ClusterResourceType.EDGE,
    devices=(num_clients // 2) * [client_0_resource],
)

edge_1_resource = ClusterResource(
    name="edge_1",
    type=ClusterResourceType.EDGE,
    devices=(num_clients // 2) * [client_1_resource],
)

exp = NetflExperiment(
    name="mnist-exp",
    task=task,
    clusters=[cloud_resource, edge_0_resource, edge_1_resource],
)

server = exp.create_server(server_resource)
edge_0_clients = exp.create_clients(client_0_resource, edge_0_resource.num_devices)
edge_1_clients = exp.create_clients(client_1_resource, edge_1_resource.num_devices)

cloud = exp.create_cluster(cloud_resource)
edge_0 = exp.create_cluster(edge_0_resource)
edge_1 = exp.create_cluster(edge_1_resource)

exp.add_to_cluster(server, cloud)

for client in edge_0_clients:
    exp.add_to_cluster(client, edge_0)
for client in edge_1_clients:
    exp.add_to_cluster(client, edge_1)

worker = exp.register_remote_worker("127.0.0.1")
worker.add_cluster(cloud)
worker.add_cluster(edge_0)
worker.add_cluster(edge_1)
worker.create_cluster_link(cloud, edge_0, NetworkResource(bw=10))
worker.create_cluster_link(cloud, edge_1, NetworkResource(bw=5))

try:
    exp.start()
except Exception as ex:
    print(ex)
finally:
    exp.stop()

```

> When the host `cpu_clock` is set to `BASE_COMPUTE_UNIT`, all resource `cpu_clock` values are interpreted in Docker CPU units (e.g., millicores) instead of GHz.

### 3. Start Fogbed Worker and Run the Experiment

```
RunWorker -p=5000
```

```
python3 experiment.py
```

Refer to the [Fogbed documentation](https://larsid.github.io/fogbed/distributed_emulation) for detailed instructions on starting workers.

## Running a Simple Example with a Basic Network Topology Using Docker Compose

### 1. Clone the repository

```
https://github.com/larsid/netfl.git
```

### 2. Create the Task

In the project root directory, create or modify a **NetFL Task** and name the file `task.py`. Refer to the examples in the `examples` folder for guidance on task creation.

### 3. Create the Infrastructure

Use Docker Compose to set up the infrastructure, including the server and clients:

```
docker compose up -d
```

### 4. View Training Results

To check the server logs, run:

```
docker logs server
```

Training logs are also stored in the logs folder within the project root directory. 

### 5. Shut Down the Infrastructure

To stop and remove all running containers, use the following command:

```
docker compose down
```

## More information

- [NetFL on GitHub](https://github.com/larsid/netfl)

- [NetFL on PyPI](https://pypi.org/project/netfl)

- [NetFL Docker Images](https://hub.docker.com/r/netfl/netfl)
