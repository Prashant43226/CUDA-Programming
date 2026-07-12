//SoftMax function CUDA
//https://leetgpu.com/challenges/softmax

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output,float sum,float maxVal, int N) 
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if (idx < N)
    output[idx]=(((expf(input[idx]-maxVal)))/(sum));
}

__global__ void sumExponent(const float * input,float* sum,float maxVal,int N){
    extern __shared__ float s[];
    int tid=threadIdx.x;
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    
    s[tid]=(idx<N)?((expf(input[idx]-maxVal))):0.0f;

    __syncthreads();

    
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            s[tid]+=s[tid+stride];
        }
        __syncthreads();
    }
    
    
    //Less efficient
    /*
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

__global__ void maxVal(const float* input,float * blockMaxes,int N){
    extern __shared__ float s[];
    int tid=threadIdx.x;
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    s[tid]=(idx<N)?(input[idx]):-FLT_MAX;
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride>>=1){
        if(tid<stride){
            s[tid]=fmaxf(s[tid],s[tid+stride]);
        }
        __syncthreads();
    }
    if(tid==0){
        blockMaxes[blockIdx.x]=s[0];
    }
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float minus_inf=-FLT_MAX;
    float* sum;
    float *dmax;
    float * blockMaxes;
    cudaMalloc(&sum,sizeof(float));
    cudaMemset(sum,0,sizeof(float));
    cudaMalloc(&dmax,sizeof(float));
    cudaMemset(dmax,minus_inf,sizeof(float));
    cudaMalloc(&blockMaxes,blocksPerGrid*sizeof(float));
    maxVal<<<blocksPerGrid,threadsPerBlock,threadsPerBlock*sizeof(float)>>>(input,blockMaxes,N);
    int currentSize=blocksPerGrid;
    while(currentSize>1){
        int newBlocks=(currentSize+threadsPerBlock-1)/threadsPerBlock;
        float* tempMaxes;
        cudaMalloc(&tempMaxes,newBlocks*sizeof(float));
        maxVal<<<newBlocks,threadsPerBlock,threadsPerBlock*sizeof(float)>>>(blockMaxes,tempMaxes,currentSize);
        currentSize=newBlocks;
        cudaFree(blockMaxes);
        blockMaxes=tempMaxes;
    }
    cudaDeviceSynchronize();
    float hMax;
    cudaMemcpy(&hMax,blockMaxes,sizeof(float),cudaMemcpyDeviceToHost);
    sumExponent<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, sum,hMax, N);
    float h_sum;
    cudaDeviceSynchronize();
    cudaMemcpy(&h_sum, sum, sizeof(float), cudaMemcpyDeviceToHost);
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output,h_sum,hMax, N);
    cudaDeviceSynchronize();
    cudaFree(sum);
}
