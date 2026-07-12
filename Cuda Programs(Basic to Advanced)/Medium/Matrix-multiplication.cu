//Matrix multiplication
//https://leetgpu.com/challenges/matrix-multiplication

#include <cuda_runtime.h>

#define TILE 16

__global__ void matrix_multiplication_kernel
(const float* A, const float* B, float* C, int M, int N,
int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row=blockIdx.y*TILE+threadIdx.y;
    int col=blockIdx.x*TILE+threadIdx.x;

    float sum=0.0f;

    for(int t=0;t<(N+TILE-1)/TILE;++t){
        int aCol=t*TILE+threadIdx.x;
        if(row<M && aCol<N){
            As[threadIdx.y][threadIdx.x]=A[row*N+aCol];
        }else{
            As[threadIdx.y][threadIdx.x]=0.0;
        }

        int bRow=t*TILE+threadIdx.y;
        if(bRow<N && col<K){
            Bs[threadIdx.y][threadIdx.x]=B[bRow*K+col];
        }else{
            Bs[threadIdx.y][threadIdx.x]=0.0;
        }

        __syncthreads();

        for(int i=0;i<TILE;++i){
            sum+=As[threadIdx.y][i]*Bs[i][threadIdx.x];
        }

        __syncthreads();

    }

    if(row<M && col<K){
        C[row*K+col]=sum;
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
