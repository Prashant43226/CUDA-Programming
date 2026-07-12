//SoftMax attention
//http://leetgpu.com/challenges/softmax-attention

#include <cuda_runtime.h>

__global__ void matrix_multiply_kernel(const float* Q, const float* K, float* QK, int M, int N, int d)
{
    // row = 0, col = 0
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N) {
        float sum = 0.0f;
        for(int i=0; i<d; i++) {
            sum += Q[row * d + i] * K[col * d + i];
        }
        QK[row * N + col] = sum / sqrtf(d);
    }
}

__global__ void find_max_kernel(float* input, float* row_max, int M, int N)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float local_max = -1e20f;
    for (int col = tid; col < N; col += blockDim.x) {
        local_max = fmaxf(local_max, input[row * N + col]);
    }

    sdata[tid] = local_max;
    __syncthreads();

    for(int s=blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid==0) {
        row_max[row] = sdata[0];
    }
}

__global__ void sum_exp_kernel(float* input, float* row_sum, float* row_max, int M, int N)
{
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    float local_sum = 0.0f;
    float mx = row_max[row];

    for (int col = tid; col < N; col += blockDim.x) {
        local_sum += expf(input[row * N + col] - mx);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for(int s=blockDim.x / 2;s > 0; s>>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        row_sum[row] = sdata[0];
    }
}

__global__ void softmax_attention_kernel(float* QK, const float* V, float* row_max, float* row_sum, float* output, int M, int N, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < M && col < d)
    {
        float val = 0.0f;
        for(int i = 0 ; i < N; i++)
        {
            float weight = expf(QK[row * N + i] - row_max[row]) / row_sum[row];
            val += weight * V[i * d + col];
        }
        output[row * d + col] = val;
    }
}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    // 1. 矩阵乘法
    float* QK;
    cudaMalloc(&QK, sizeof(float) * M * N);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1)/ threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(Q, K, QK, M, N, d);
    cudaDeviceSynchronize();

    // 2. 找每行的最大值
    float* d_row_max;
    cudaMalloc(&d_row_max, sizeof(float) * M);
    int threadsPerBlock1D = 256;

    size_t sharedSize = threadsPerBlock1D * sizeof(float);

    find_max_kernel<<<M, threadsPerBlock1D, sharedSize>>>(QK, d_row_max, M, N);
    cudaDeviceSynchronize();

    // 3. 计算每行的累加和
    float* d_row_sum;
    cudaMalloc(&d_row_sum, sizeof(float) * M);
    sum_exp_kernel<<<M, threadsPerBlock1D, sharedSize>>>(QK, d_row_sum, d_row_max, M, N);
    cudaDeviceSynchronize();

    // 4. 计算softmax attention
    dim3 blocksPerGrid2D((d + threadsPerBlock.x - 1)/ threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    softmax_attention_kernel<<<blocksPerGrid2D, threadsPerBlock>>>(QK, V, d_row_max, d_row_sum, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(QK);
    cudaFree(d_row_max);
    cudaFree(d_row_sum);
}
