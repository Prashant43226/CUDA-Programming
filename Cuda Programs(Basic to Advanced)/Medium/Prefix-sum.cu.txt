//Prefix Sum(nlogn)
//https://leetgpu.com/challenges/prefix-sum


#include <cuda_runtime.h>
#include<bits/stdc++.h>
#include<stdio.h>

__global__ void prefSum(const float * input,float * output,float * tempSumArr,int N){
    extern __shared__ float s[];
    int tid=threadIdx.x;
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    float val=(idx<N)?input[idx]:0.0f;
    s[tid]=val;
    __syncthreads();

    for(int stride=1;stride<blockDim.x;stride*=2){
        float temp=s[tid];
        if(tid>=stride){
            temp+=s[tid-stride];
        }
        __syncthreads();
        s[tid]=temp;
        __syncthreads();
    }
    if(idx<N){
        output[idx]=s[tid];
    }
    int last = min(blockDim.x, N - blockIdx.x*blockDim.x) - 1;
    if(tempSumArr!=nullptr && tid==last){
        tempSumArr[blockIdx.x]=s[tid];
    }

}

__global__ void addBlockSum(float * output,float * tempSumArr,int N){
    int tid=threadIdx.x;
    int gid=blockDim.x*blockIdx.x+tid;
    if(gid<N){
        if(blockIdx.x>0)
        output[gid]+=tempSumArr[blockIdx.x-1];
    }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock=256;
    int blocksPerGrid=(N+threadsPerBlock-1)/threadsPerBlock;
    float * tempSumArr;
    cudaMalloc(&tempSumArr,blocksPerGrid*sizeof(float));
    prefSum<<<blocksPerGrid,threadsPerBlock,threadsPerBlock*sizeof(float)>>>(input,output,tempSumArr,N);
    if(blocksPerGrid>1){
        solve(tempSumArr,tempSumArr,blocksPerGrid);
        addBlockSum<<<blocksPerGrid,threadsPerBlock>>>(output,tempSumArr,N);
    }
    cudaFree(tempSumArr);
}
