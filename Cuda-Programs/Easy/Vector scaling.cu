//Vector scaling problem
#include<stdio.h>
#include<cuda.h>

__global__ void scale(int * a,int * c,int k,int N){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N) //Important to add this statement else accesses invalid memory locations though result shows up fine
  c[idx]=a[idx]*k;
}
int main(){
  int N=1000;
  int a[N];
  int C[N];
  int k=5;
  int * da,*dc;
  int threads=256;
  int blocks=(N+threads-1)/threads;

  for(int i=0;i<1000;++i){
    a[i]=i;
  }


  cudaMalloc(&da,N*sizeof(int));
  cudaMalloc(&dc,N*sizeof(int));

  cudaMemcpy(da,a,N*sizeof(int),cudaMemcpyHostToDevice);

  scale<<<blocks,threads>>>(da,dc,k,N);

  cudaMemcpy(C,dc,N*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i=0;i<N;++i){
    printf("%d ",C[i]);
  }

  cudaFree(da);
  cudaFree(dc);
}
