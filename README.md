## matmul0

General comparison between:

- CPU matrix multiplication
- GPU global memory matrix multiplication
- GPU shared memory matrix multiplication


---

## matmul1


### 1. General Shared Memory Matrix Multiplication

Analysis of a standard tiled shared memory matrix multiplication kernel.

### 2. Shared Memory Matrix Multiplication

Configuration:

- Tile size = 4 × block size
- Coarsening factor = 1

This implementation studies the effect of increasing tile size while keeping the coarsening factor minimal.

### 3. Shared Memory Matrix Multiplication

Configuration:

- Tile size = 4 × block size
- Coarsening factor = 4

This implementation analyzes the impact of thread coarsening on performance and resource utilization.

---

## neural_networks

CUDA-based neural network implementation trained on the MNIST dataset.

includes:

- Forward propagation
- Training using gradient-based optimization
- Accuracy and loss tracking
- MNIST dataset classification

