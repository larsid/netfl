# Experiment Specifications

  - Server (2.0 GHz, 2 GB, 1 Gbps)
  - Dataset: CIFAR-10
  - Train Size: 50,000
  - Test Size: 10,000
  - Model: CNN3
  - Optimizer: SGD
  - Learning Rate: 0.01
  - Aggregation Function: FedAvg
  - Batch Size: 16
  - Epochs: 2
  - Rounds: 500

## 1. Resources

### 1.1 Device Allocation

#### 1.1.1

  - Devices: 8 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Bandwidth: 100 Mbps
  - Partitioning: IID

#### 1.1.2

  - Devices: 16 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Bandwidth: 100 Mbps
  - Partitioning: IID

#### 1.1.3

  - Devices: 32 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Bandwidth: 100 Mbps
  - Partitioning: IID

#### 1.1.4

  - Devices: 64 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Bandwidth: 100 Mbps
  - Partitioning: IID

### 1.2 Network Bandwidth

#### 1.2.1

  - Devices: 32 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Bandwidth: 50 Mbps
  - Partitioning: IID

#### 1.2.2

  - Devices: 32 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Bandwidth: 25 Mbps
  - Partitioning: IID

## 2. Heterogeneity

### 2.1 Device Heterogeneity
 
#### 2.1.1

  - Devices: 16 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Devices: 16 × Raspberry Pi 4 (1.5 GHz, 4 GB)
  - Bandwidth: 100 Mbps (Raspberry Pi 3)
  - Bandwidth: 1 Gbps (Raspberry Pi 4)
  - Partitioning: IID

#### 2.1.2

  - Devices: 32 × Raspberry Pi 4 (1.5 GHz, 4 GB)
  - Bandwidth: 1 Gbps (Raspberry Pi 4)
  - Partitioning: IID

### 2.2 Data Heterogeneity

#### 2.2.1

  - Devices: 32 × Raspberry Pi 4 (1.5 GHz, 4 GB)
  - Bandwidth: 1 Gbps
  - Partitioning: Dirichlet (α = 1.0, balanced, min size = 781)

#### 2.2.2
  
  - Devices: 32 × Raspberry Pi 4 (1.5 GHz, 4 GB)
  - Bandwidth: 1 Gbps
  - Partitioning: Dirichlet (α = 0.01, balanced, min size = 781)
