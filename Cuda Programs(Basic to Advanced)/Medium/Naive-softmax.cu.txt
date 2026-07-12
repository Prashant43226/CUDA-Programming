//Naive SoftMax
//https://leetgpu.com/challenges/softmax

#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output,float sum, int N) 
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx < N)
    output[idx]=(((expf(input[idx])))/(sum));
}

__global__ void sumExponent(const float * input,float* sum,int N){
    extern __shared__ float s[];
    int tid=threadIdx.x;
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    
    s[tid]=(idx<N)?((expf(input[idx]))):0.0f;

    __syncthreads();

    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            s[tid]+=s[tid+stride];
        }
        __syncthreads();
    }
    /*
    Less efficient
    for(int stride=1;stride<blockDim.x;stride*=2){
        int id=2*stride*tid;
        if(id+stride<blockDim.x){
            s[id]+=s[id+stride];
        }
        __syncthreads();
    }*/

    if(tid==0){
        atomicAdd(sum,s[0]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* sum;
    cudaMalloc(&sum,sizeof(float));
    cudaMemset(&sum,0,sizeof(float));
    sumExponent<<<blocksPerGrid, threadsPerBlock,threadsPerBlock * sizeof(float)>>>(input, sum, N);
    float h_sum;
    cudaMemcpy(&h_sum, sum, sizeof(float), cudaMemcpyDeviceToHost);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output,h_sum, N);
    cudaDeviceSynchronize();
    cudaFree(sum);
}
