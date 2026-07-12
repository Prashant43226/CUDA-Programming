//Simple histogramming
//https://leetgpu.com/challenges/histogramming

#include <cuda_runtime.h>

__global__ void hist(const int * input,int * histogram,int N){
    int idx=threadIdx.x;
    int tid=blockDim.x*blockIdx.x+threadIdx.x;
    int val=-1;
    if(tid<N){
        val=input[tid];
    }
    if(val>=0){
        atomicAdd(histogram+val,1);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    int threadsPerBlock=256;
    int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;
    hist<<<blocksPerGrid,threadsPerBlock>>>(input,histogram,N);
}
