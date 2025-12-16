# Configurations

## 1. Hardware and Training

- Host: 8-Core 2.25 GHz, 64 GB
- Dataset: CIFAR-10 (Train size: 50000 / Test size: 10000)
- Partitions: 64
- Model: CNN3
- Optimizer: SGD (Learning rate: 0.01)
- Aggregation Function: FedAvg
- Batch Size: 16
- Local Epochs: 2
- Global Rounds: 500

## 2. Evaluation Metrics

| **Metric**               | **Unit**   | **Description**                                                                                                                            |
| ------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Accuracy                 | Percentage | Represents the performance of the global model, measured as the percentage of correctly classified samples on the server’s test dataset.   |
| Convergence Speed        | No. Rounds | Indicates how many communication rounds are required for the global model to reach a stable or optimal accuracy level.                     |
| Avg Training Time        | Second     | Mean time spent by clients on local model training during each round, averaged across all participating devices.                           |
| Avg Memory Utilization   | MB         | Average portion of memory resources used by clients while performing local training operations.                                            |
| Avg CPU Utilization      | Percentage | Mean CPU resource usage observed during client-side training processes.                                                                    |
| Avg Update Exchange Time | Second     | Average duration of the communication cycle in which clients send their model updates to the server and receive the aggregated model back. |

## 3. Data Partitioning

### 3.1 IID

IID partitioner:

  <img src="https://i.postimg.cc/ryR0mMS8/CIFAR10-IID.png" width="400">

### 3.2 Non-IID

Pathological partitioner with:

- Classes per partition: 4
- Class assignment mode: deterministic

  <img src="https://i.postimg.cc/zff31rzy/CIFAR10-Non-IID.png" width="400">

### 3.3 Extreme Non-IID

Pathological partitioner with:

- Classes per partition: 1
- Class assignment mode: deterministic

  <img src="https://i.postimg.cc/FRdR7VJt/CIFAR10-Extreme-Non-IID.png" width="400">

# Experiments

- Server: 14-Core 2.0 GHz, 8 GB, 1 Gbps
- Raspberry Pi 3: 4-Core 1.2 GHz, 1 GB, 100 Mbps
- Raspberry Pi 4: 4-Core 1.5 GHz, 4 GB, 1 Gbps

## 1 Device Allocation

Investigate the trade-offs of increasing the number of participating clients in a homogeneous environment. The objective is to observe two correlations:

- The positive correlation between the number of clients (from 8 to 32) and the final Test Accuracy, which is expected to improve as more total data is involved in each federated round.
- The impact of client scaling on operational costs, specifically the Avg Update Exchange Time, to measure the cost of increased aggregation and network overhead.

### 1.1

- Devices: 8 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: IID

### 1.2

- Devices: 16 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: IID

### 1.3

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: IID

### 1.4

- Devices: 64 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: IID

## 2 Network Bandwidth

Investigate the impact of communication constraints on operational costs. The objective is to measure the relationship between network bandwidth and training delays:

- Quantify the change in Avg Update Exchange Time when bandwidth is limited to 25 Mbps, 50 Mbps, and 100 Mbps.
- Confirm that communication delays become a significant bottleneck as available bandwidth is restricted.

### 2.1

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 25 Mbps
- Partitioning: IID

### 2.2

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 50 Mbps
- Partitioning: IID

### 2.3 (same as 1.3)

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: IID

## 3 Data Heterogeneity

Investigate the impact of statistical data heterogeneity on the global model's performance. The objective is to compare the model's effectiveness under different data distributions:

- Compare the final Test Accuracy and Convergence Speed of models trained on IID, Non-IID, and Extreme Non-IID data.
- Demonstrate the progressive degradation in model accuracy as the degree of data skew increases.

### 3.1 (same as 1.3)

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: IID

### 3.2

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: Non-IID

### 3.3

- Devices: 32 × Raspberry Pi 3
- Link Bandwidth: 100 Mbps
- Partitioning: Extreme Non-IID

## 4 Device Heterogeneity

Investigate the operational inefficiencies caused by hardware heterogeneity in a federation. The objective is to measure the straggler effect in a mixed-device environment:

- Observe the performance gap by comparing the Avg Training Time of low-performance devices versus high-performance devices.
- Demonstrate the system bottleneck by measuring the Avg Update Exchange Time for high-performance devices, which is expected to be high due to idle time spent waiting for stragglers to complete their local training.

### 4.1

- Devices: 16 × Raspberry Pi 3 and 16 × Raspberry Pi 4
- Link Bandwidth: 100 Mbps
- Partitioning: IID
