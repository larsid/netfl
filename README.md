# FogFL

**FogFL** is a plugin that extends **Fogbed** by integrating the **Flower** framework, enabling the execution of Federated Learning experiments in Fog/Edge environments. This integration allows seamless deployment and management of distributed machine learning tasks across edge and fog nodes, enhancing the capabilities of **Fogbed** for federated learning simulations.

## How to Run Distributed Experiment

To run a distributed Federated Learning experiment using FogFL, Fogbed will automatically create and manage the distributed infrastructure for you. This involves setting up multiple clients and a server across different nodes or containers, simulating a distributed environment. Fogbed takes care of the orchestration, so you can focus on running and monitoring the experiment. Follow the instructions to start the distributed experiment:

## How to Run Locally with Docker

1. **Create the Main Task**  
   In the project root directory, create an FL Task and name the file `task.py`. You can refer to the example in the `examples/task` folder for guidance.

2. **Build the Docker Image**  
   In the project root directory, run the following command to build the Docker image:
   ```
   docker build -t fogfl:local .
   ```

3. **Create the Infrastructure**  
   In the project root directory, use Docker Compose to start both the server and the clients:
   ```
   docker compose up -d
   ```

4. **Start Clients**  
   - Enter the container of **client 0** and start the lazy client:
     ```
     docker exec -it client_0 bash
     python run.py --type=client --server_port=8181 --server_address="172.18.0.2" --client_id=0 --lazy_client=true
     ```
   
   - Enter the container of **client 1** and start the lazy client:
     ```
     docker exec -it client_1 bash
     python run.py --type=client --server_port=8181 --server_address="172.18.0.2" --client_id=1 --lazy_client=true
     ```

5. **Start the Server**  
   - Enter the container of the **server** and run it:
     ```
     docker exec -it server bash
     python run.py --type=server --server_port=8181
     ```

6. **Shut Down the Infrastructure**  
   To stop and remove all running containers, in the project root directory, use the following Docker Compose command:
   ```
   docker compose down
   ```

7. **Remove the Docker Image**  
   Run the following command to remove the local Docker image created:
   ```
   docker rmi fogfl:local
   ```
