# NetFL

**NetFL** is a plugin that extends **Fogbed** by integrating the **Flower** framework, enabling the execution of Federated Learning experiments in Fog/Edge environments. This integration allows seamless deployment and management of distributed machine learning tasks across edge and fog nodes, enhancing **Fogbed's** capabilities for federated learning simulations.

## Running an Experiment with NetFL and Fogbed

NetFL simplifies the execution of Federated Learning experiments by automatically setting up and managing the required infrastructure. This includes configuring multiple clients and a central server across different nodes or containers, simulating a distributed environment. Fogbed handles the orchestration, allowing you to focus on running and monitoring your experiment.

Follow the steps below to set up and run an experiment using **NetFL**. This is an example using the **MNIST** dataset. You can find more examples in the `examples` folder:

### 1. Define the Dataset, the Model, and the Training Configurations

```py
from keras import layers, models
from flwr.server.strategy import Strategy, FedAvg

from netfl.core.task import Dataset, Task, TrainConfig, DatasetInfo


class MNIST(Task):
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            huggingface_path="ylecun/mnist",
            item_name="image",
            label_name="label",
        )

    def dataset(self, raw_dataset: Dataset) -> Dataset:
        normalized_dataset = Dataset(
            x_train=(raw_dataset.x_train / 255.0),
            x_test=(raw_dataset.x_test / 255.0),
            y_train=raw_dataset.y_train,
            y_test=raw_dataset.y_test,
        )
        return normalized_dataset

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

    def aggregation_strategy(self) -> Strategy:
        return self._aggregation_strategy_factory(FedAvg)
    
    def train_config(self) -> TrainConfig:
	    return TrainConfig(
            batch_size=32,
            epochs=1,
            fraction_evaluate=1.0,
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

### 2. Start a Fogbed Worker and Define the Network Topology

Refer to the **Fogbed documentation** for instructions on [installation](https://larsid.github.io/fogbed/#install) and [starting the worker](https://larsid.github.io/fogbed/distributed_emulation).

<p align="center">
  <img src="examples/mnist/network-topology.png" alt="Network Topology" width="500"/>
</p>

### 3. Create and Run the Experiment

```py
from netfl.infra.experiment import Experiment
from task import MainTask

exp = Experiment(
    main_task=MainTask()
)

worker_0 = exp.add_worker(ip="worker-ip", port="worker-port")

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

```

## Running Locally with Docker

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
