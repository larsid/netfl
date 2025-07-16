# Experiment Specifications

  - Dataset: CIFAR-10
  - Train Size: 50,000
  - Test Size: 10,000
  - Model: CNN3
  - Optimizer: SGD
  - Learning Rate: 0.01
  - Aggreation Function: FedAvg
  - Batch Size: 16
  - Epochs: 2
  - Rounds: 500

## 1. Resources

### 1.1 Device Allocation

  - Bandwidth: 100 Mbps
  - Partitioning: IID

#### 1.1.1

  - Devices: 8 × Raspberry Pi 3 (1.2 GHz, 1 GB)

#### 1.1.2

  - Devices: 16 × Raspberry Pi 3 (1.2 GHz, 1 GB)

#### 1.1.3

  - Devices: 32 × Raspberry Pi 3 (1.2 GHz, 1 GB)

#### 1.1.4

  - Devices: 64 × Raspberry Pi 3 (1.2 GHz, 1 GB)

### 1.2 Network Bandwidth

  - Devices: 32 × Raspberry Pi 3 (1.2 GHz, 1 GB)
  - Partitioning: IID

#### 1.2.1

  - Bandwidth: 50 Mbps

#### 1.2.2

  - Bandwidth: 25 Mbps

## 2. Heterogeneity

### 2.1 Device Heterogeneity

  - Bandwidth: 100 Mbps
  - Partitioning: IID
 
#### 2.1.1

  - Devices: 16 × Raspberry Pi 3 (1.2 GHz, 1 GB) and 16 × Raspberry Pi 4 (1.5 GHz, 4 GB)

#### 2.1.2

  - Devices: 32 × Raspberry Pi 4 (1.5 GHz, 4 GB)

### 2.2 Data Heterogeneity

  - Devices: 32 × Raspberry Pi 4 (1.5 GHz, 4 GB)
  - Bandwidth: 1 Gbps

#### 2.2.1

  - Partitioning: Non-IID (Dirichlet with α = 1.0)

#### 2.2.2
  
  - Partitioning: ExNon-IID (Dirichlet with α = 0.01)
