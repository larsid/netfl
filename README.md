# NetFL

**NetFL** is a framework that extends [Fogbed](https://github.com/larsid/fogbed) by integrating [Flower](https://github.com/adap/flower), enabling simulation of Federated Learning experiments within Fog/Edge computing environments. It supports the modeling of heterogeneous and resource-constrained edge scenarios, incorporating factors such as computational disparities among devices and dynamic network conditions, including bandwidth limitations, latency variations, and packet loss. This facilitates realistic evaluations of FL systems under non-ideal, real-world conditions.

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
			label_dtype=tf.int32
		)
	
	def dataset_partitioner(self) -> DatasetPartitioner:
		return IidPartitioner()

	def preprocess_dataset(self, dataset: Dataset, training: bool) -> Dataset:
		return Dataset(
			x=tf.divide(dataset.x, 255.0),
			y=dataset.y
		)

	def model(self) -> models.Model:        
		return cnn3(
			input_shape=(28, 28, 1),
			output_classes=10,
			optimizer=optimizers.SGD(learning_rate=0.01)
		)

	def aggregation_strategy(self) -> tuple[type[Strategy], dict[str, Any]]:
		return FedAvg, {}
	
	def train_configs(self) -> TrainConfigs:
		return TrainConfigs(
			batch_size=16,
			epochs=2,
			num_devices=4,
			num_partitions=4,
			num_rounds=10,
			seed_data=42,
			shuffle_data=True
		)


class MainTask(MNIST):
	pass

```

### 2. Define the Experiment

![Network Topology](https://i.postimg.cc/3r2k2W90/network-topology.png)


```py
from netfl.core.experiment import NetflExperiment
from netfl.utils.resources import Host, NetworkResource, Resource, ClusterResource, ClusterResourceType

from task import MainTask


task = MainTask()
train_configs = task.train_configs()

host = Host()

server_resource = Resource(
	name="server",
	cpu_cores=1,
	cpu_clock=1.0,
	memory=1024,
	network=NetworkResource(bw=1000),
	host=host
)

device_0_resource = Resource(
	name="device_0",
	cpu_cores=1,
	cpu_clock=0.25,
	memory=512,
	network=NetworkResource(bw=100),
	host=host
)

device_1_resource = Resource(
	name="device_1",
	cpu_cores=1,
	cpu_clock=0.25,
	memory=512,
	network=NetworkResource(bw=50),
	host=host
)

cloud_resource = ClusterResource(
	name="cloud",
	type=ClusterResourceType.CLOUD,
	resources=[server_resource]
)

edge_0_resource = ClusterResource(
	name="edge_0",
	type=ClusterResourceType.EDGE,
	resources=(train_configs.num_devices // 2) * [device_0_resource]
)

edge_1_resource = ClusterResource(
	name="edge_1",
	type=ClusterResourceType.EDGE,
	resources=(train_configs.num_devices // 2) * [device_1_resource]
)

exp = NetflExperiment(
	name="mnist-exp", 
	task=task, 
	resources=[cloud_resource, edge_0_resource, edge_1_resource]
)

cloud = exp.create_cluster(cloud_resource)
edge_0 = exp.create_cluster(edge_0_resource)
edge_1 = exp.create_cluster(edge_1_resource)

server = exp.create_server(server_resource)
edge_0_devices = exp.create_devices(device_0_resource, edge_0_resource.num_resources)
edge_1_devices = exp.create_devices(device_1_resource, edge_1_resource.num_resources)

exp.add_to_cluster(server, cloud)
for device in edge_0_devices: exp.add_to_cluster(device, edge_0)
for device in edge_1_devices: exp.add_to_cluster(device, edge_1)

worker = exp.add_worker("127.0.0.1")
worker.add(cloud)
worker.add(edge_0)
worker.add(edge_1)
worker.add_link(cloud, edge_0, **NetworkResource(bw=10).link_params)
worker.add_link(cloud, edge_1, **NetworkResource(bw=5).link_params)

try:
	exp.start()
except Exception as ex: 
	print(ex)
finally:
	exp.stop()

```

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

- [NetFL on PyPI](https://pypi.org/project/netfl)

- [NetFL Docker Images](https://hub.docker.com/r/netfl/netfl/tags)
