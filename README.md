# CUDA Programming

**Source of Challenges:**  
https://leetgpu.com/challenges

## Overview

This repository contains CUDA kernel implementations of various programs ranging from **Basic** to **Advanced**. The goal is to learn different CUDA programming concepts, parallel programming techniques, GPU memory hierarchy, and optimization strategies through hands-on implementations.

The primary objective of this repository is to understand how CUDA kernels are implemented for a variety of computational problems while documenting my learning journey. These programs demonstrate many of the fundamental building blocks used in modern AI systems at a low level. Although modern deep learning libraries abstract away these implementations and provide highly optimized kernels, understanding the underlying CUDA concepts is invaluable for learning GPU programming and performance optimization.

---

# Programs Covered

## 1. Basic

| # | Program | Solution |
|---|---------|----------|
| 1.1 | Print Block Size in CUDA | [Solution](./Cuda-Programs/Basic/Print-Block-size.cu) |
| 1.2 | Print Threads in CUDA | [Solution](./Cuda-Programs/Basic/Print-threads.cu) |

---

## 2. Easy

| # | Program | Challenge | Solution |
|---|---------|-----------|----------|
| 2.1 | Add Two Arrays in Parallel using CUDA |  | [Solution](./Cuda-Programs/Easy/Program-to-add-2-arrays-parallely-in-CUDA.cu) |
| 2.2 | Add Two Vectors of Different Sizes using CUDA | https://leetgpu.com/challenges/vector-addition | [Solution](./Cuda-Programs/Easy/Program-to-add-2-vectors-with-different-block-size-and-threadpool-size.cu) |
| 2.3 | Vector Addition and Vector Scaling | - | [Solution](./Cuda-Programs/Easy/Vector-addition.cu) |

---

## 3. Medium

| # | Program | Challenge | Solution |
|---|---------|-----------|----------|
| 3.1 | Reduction in CUDA | https://leetgpu.com/challenges/reduction | [Solution](./Cuda-Programs/Medium/Reduction.cu) |
| 3.2 | Softmax in CUDA | https://leetgpu.com/challenges/softmax | [Solution](./Cuda-Programs/Medium/Softmax.cu) |
| 3.3 | 1D Convolution in CUDA | https://leetgpu.com/challenges/1d-convolution | [Solution](./Cuda-Programs/Medium/1D-convolution.cu) |
| 3.4 | Histogramming in CUDA | https://leetgpu.com/challenges/histogramming | [Solution](./Cuda-Programs/Medium/Histogram.cu) |
| 3.5 | Matrix Multiplication in CUDA | https://leetgpu.com/challenges/matrix-multiplication | [Solution](./Cuda-Programs/Medium/Matrix-multiplication.cu/) |
| 3.6 | Matrix Transpose in CUDA | https://leetgpu.com/challenges/matrix-transpose | [Solution](./Cuda-Programs/Medium/Matrix-transpose.cu) |
| 3.7 | Prefix Sum (Scan) in CUDA | https://leetgpu.com/challenges/prefix-sum | [Solution](./Cuda-Programs/Medium/Prefix-sum.cu/) |
| 3.8 | Softmax Attention in CUDA | https://leetgpu.com/challenges/softmax-attention | [Solution](./Cuda-Programs/Medium/softmax-attention.cu/) |

---

# Topics Covered

- CUDA Programming Fundamentals
- CUDA Execution Model (Grids, Blocks, and Threads)
- Global, Shared, and Constant Memory
- Synchronization using `__syncthreads()`
- Parallel Reduction
- Parallel Prefix Sum (Scan)
- Histogramming
- Matrix Operations
- Convolution
- Softmax and Attention
- Memory Optimization Techniques
- Parallel Programming Patterns

---

# Repository Structure

```text
CUDA-Programming/
│
├── Basic/
├── Easy/
├── Medium/
├── Hard/
└── README.md
```

---

# Learning Objectives

- Understand CUDA programming from the ground up.
- Learn how GPU kernels execute in parallel.
- Explore CUDA memory hierarchy and synchronization primitives.
- Implement common parallel algorithms from scratch.
- Gain intuition behind low-level implementations used in modern AI frameworks.
- Build a strong foundation for CUDA optimization and GPU computing.

---

# References

- **LeetGPU:** https://leetgpu.com/challenges
- **Lecture Series:(GPU Programming by Dr.Rupesh Nasre IIT Madras)** https://youtu.be/wFlejBXX9Gk?si=MwdZzhsFW-rTEXYQ
- **NVIDIA CUDA Documentation:** https://docs.nvidia.com/cuda/
- **CUDA C++ Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

⭐ If you find this repository useful, consider giving it a star!
