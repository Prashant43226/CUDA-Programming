//Reduction
//https://leetgpu.com/challenges/reduction

#include <cuda_runtime.h>
#include<stdio.h>

// input, output are device pointers
__global__ void reduce(const float * input,float * output,int N){
    extern __shared__ float sum[];

    int tid=threadIdx.x;
    int idx=blockDim.x*blockIdx.x+threadIdx.x;

    sum[tid]=(idx<N)?input[idx]:0.0f;

    __syncthreads();

    /*Method 1:
    for(int stride=blockDim.x/2;stride>0;stride=stride>>1){
        if(tid<stride){
        sum[tid]+=sum[tid+stride];
        }
         __syncthreads();
    }*/
    
    /*Method 2*/
    for(int stride=1;stride<blockDim.x;stride*=2){
        int id=2*stride*tid;
        if(id+stride<blockDim.x){
            sum[id]+=sum[id+stride];
        }
        __syncthreads();
    }

    if(tid==0){
        output[blockIdx.x]=sum[0];
    }

}
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock=256;
    int blockSize=(N+threadsPerBlock-1)/threadsPerBlock;
    reduce<<<blockSize,threadsPerBlock,threadsPerBlock*sizeof(float)>>>(input,output,N);
}
