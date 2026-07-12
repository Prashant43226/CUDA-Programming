# CUDA Programming

**Source of Challenges:**  
https://leetgpu.com/challenges

## Overview

This repository contains raw CUDA implementations of various programs ranging from **Basic** to **Advanced**. The goal is to learn different CUDA programming concepts, parallel programming techniques, GPU memory hierarchy, and optimization strategies through hands-on implementations.

The primary objective of this repository is to understand how CUDA kernels are implemented for a variety of computational problems while documenting my learning journey. These programs demonstrate many of the fundamental building blocks used in modern AI systems at a low level. Although modern deep learning libraries abstract away these implementations and provide highly optimized kernels, understanding the underlying CUDA concepts is invaluable for learning GPU programming and performance optimization.

---

# Programs Covered

## 1. Basic

| # | Program | Solution |
|---|---------|----------|
| 1.1 | Print Block Size in CUDA | [Solution](./Basic) |
| 1.2 | Print Threads in CUDA | [Solution](./Basic/Print-Threads/) |

---
Cuda Programs(Basic to Advanced)/Basic/Print-Block-size.cu

## 2. Easy

| # | Program | Challenge | Solution |
|---|---------|-----------|----------|
| 2.1 | Add Two Arrays in Parallel using CUDA |  | [Solution](./Easy/Array-Addition/) |
| 2.2 | Add Two Vectors of Different Sizes using CUDA | https://leetgpu.com/challenges/vector-addition | [Solution](./Easy/Vector-Addition/) |
| 2.3 | Vector Addition and Vector Scaling | - | [Solution](./Easy/Vector-Scaling/) |

---

## 3. Medium

| # | Program | Challenge | Solution |
|---|---------|-----------|----------|
| 3.1 | Reduction in CUDA | https://leetgpu.com/challenges/reduction | [Solution](./Medium/Reduction/) |
| 3.2 | Softmax in CUDA | https://leetgpu.com/challenges/softmax | [Solution](./Medium/Softmax/) |
| 3.3 | 1D Convolution in CUDA | https://leetgpu.com/challenges/1d-convolution | [Solution](./Medium/1D-Convolution/) |
| 3.4 | Histogramming in CUDA | https://leetgpu.com/challenges/histogramming | [Solution](./Medium/Histogramming/) |
| 3.5 | Matrix Multiplication in CUDA | https://leetgpu.com/challenges/matrix-multiplication | [Solution](./Medium/Matrix-Multiplication/) |
| 3.6 | Matrix Transpose in CUDA | https://leetgpu.com/challenges/matrix-transpose | [Solution](./Medium/Matrix-Transpose/) |
| 3.7 | Prefix Sum (Scan) in CUDA | https://leetgpu.com/challenges/prefix-sum | [Solution](./Medium/Prefix-Sum/) |
| 3.8 | Softmax Attention in CUDA | https://leetgpu.com/challenges/softmax-attention | [Solution](./Medium/Softmax-Attention/) |

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
