//Matrix transpose 
//https://leetgpu.com/challenges/matrix-transpose

#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_transpose_kernel(const float *input,float *output,int rows,int cols)
{
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x; 
    int y = blockIdx.y * TILE + threadIdx.y; 

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < rows && y < cols)
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
