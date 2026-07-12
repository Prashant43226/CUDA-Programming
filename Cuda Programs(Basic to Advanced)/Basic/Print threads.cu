//Program to check the thread index and how it behaves when Blocksize is more than warp size

#include<stdio.h>
#include<cuda.h>
__global__ void hello(){
    int idx=threadIdx.x;
    printf("Printing thread idx%d",idx);
}
int main(){
    hello<<<1,1000>>>();
    cudaDeviceSynchronize();
    return 0;
}
