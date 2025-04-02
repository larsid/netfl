# FogFL

**FogFL** is a plugin that extends **Fogbed** by integrating the **Flower** framework, enabling the execution of Federated Learning experiments in Fog/Edge environments. This integration allows seamless deployment and management of distributed machine learning tasks across edge and fog nodes, enhancing **Fogbed's** capabilities for federated learning simulations.

## Running a Distributed Experiment

FogFL simplifies the execution of distributed Federated Learning experiments by automatically setting up and managing the required infrastructure. This includes configuring multiple clients and a central server across different nodes or containers, simulating a distributed environment. Fogbed handles the orchestration, allowing you to focus on running and monitoring your experiment.

Follow the steps below to set up and run a distributed experiment using **FogFL**:

## Running Locally with Docker

### 1. Create the Main Task

In the project root directory, create or modify a **FogFL Task** and name the file `task.py`. Refer to the examples in the `examples/tasks` folder for guidance on task creation.

### 2. Build the Docker Image

Run the following command in the project root directory to build the Docker image:

```
docker build -t fogfl:local .
```

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

Training logs are also stored in log files within the project root directory.

### 5. Shut Down the Infrastructure

To stop and remove all running containers, use the following command:

```
docker compose down
```

### 6. Remove the Docker Image (Optional)

If you need to remove the locally created Docker image, run:

```
docker rmi fogfl:local
```
