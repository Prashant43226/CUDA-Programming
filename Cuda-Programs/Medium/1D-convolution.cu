//1D convolution of input vector with another vector
//https://leetgpu.com/challenges/1d-convolution

#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) 
{ 
    extern __shared__ float s[];   
    int tx=threadIdx.x;
    int gid=blockDim.x*blockIdx.x+tx;
    int N=input_size;
    int K=kernel_size;
    if(gid<N){
        s[tx]=input[gid];
    }
    /*if(tx<K-1){
        int idx = min(gid + blockDim.x, N - 1);
        s[blockDim.x + tx] = input[idx];
    }*/
    for (int i = tx; i < K - 1; i += blockDim.x) {
        int idx = min(blockIdx.x * blockDim.x + blockDim.x + i, N - 1);
        s[blockDim.x + i] = input[idx];
    }
    __syncthreads();
    if(gid<N-K+1){
        float sum=0.0f;
        for(int k=0;k<K;++k)
            sum+=s[tx+k]*kernel[k];
        output[gid]=sum;        
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedSize = (threadsPerBlock + kernel_size - 1) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock,sharedSize>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
