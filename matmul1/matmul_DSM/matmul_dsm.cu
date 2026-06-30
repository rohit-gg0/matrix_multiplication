#include<cuda_runtime.h>
#include<cooperative_groups.h>
#include<stdio.h>
#include<iostream>

namespace cg = cooperative_groups;

__global__ __cluster_dims__(2,1,1) void matmul_kernel(float* A,float* B,float* C,int na,int ma,int nb,int mb,int tile_dim)
{
    extern __shared__ float shmem[];
    float* sha = shmem;
    float* shb = shmem + tile_dim*tile_dim;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int tiy = blockIdx.y*blockDim.y + ty;
    int tix = blockIdx.x*blockDim.x + tx;

    float oc = 0.0f;

    cg::cluster_group cluster = cg::this_cluster();
    int rank = cluster.block_rank();

    int phases = (ma + tile_dim - 1)/tile_dim;

    for(int m=0;m<phases;m++)
    {
        if(tiy<na && (m*tile_dim + tx)<ma && rank==0) sha[ty*tile_dim+tx] = A[tiy*ma + (m*tile_dim + tx)];
        else sha[ty*tile_dim+tx] = 0.0f;
        if(tix<mb && (m*tile_dim + ty)<nb) shb[ty*tile_dim+tx] = B[(m*tile_dim + ty)*mb + tix];
        else shb[ty*tile_dim+tx] = 0.0f;

        cluster.sync();

        float* csa = cluster.map_shared_rank(sha,0);

        for(int k=0;k<tile_dim;k++) oc += csa[ty*tile_dim+k]*shb[k*tile_dim+tx];

        cluster.sync();
    }

    if(tiy < na && tix<mb) C[tiy*mb + tix] = oc;
}

void matmul(float* A,float* B,float* C,int na,int ma,int nb,int mb,int tile_dim)
{
    if(ma!=nb)
    {
        printf("Not possibee\n");
        return;
    }

    float* dA;
    float* dB;
    float* dC;

    cudaMalloc(&dA,na*ma*sizeof(float));
    cudaMalloc(&dB,nb*mb*sizeof(float));
    cudaMalloc(&dC,na*mb*sizeof(float));

    cudaMemcpy(dA,A,na*ma*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,B,nb*mb*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block_size(tile_dim,tile_dim);
    dim3 grid_size((mb+tile_dim-1)/tile_dim,(na+tile_dim-1)/tile_dim);

    int shmem = 2*tile_dim*tile_dim*sizeof(float);

    matmul_kernel<<<grid_size,block_size,shmem>>>(dA,dB,dC,na,ma,nb,mb,tile_dim); 

    cudaMemcpy(C,dC,na*mb*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char* argv[])
{
    int na = atoi(argv[1]);
    int ma = atoi(argv[2]);
    int nb = atoi(argv[3]);
    int mb = atoi(argv[4]);

    float* A = (float*)malloc(na*ma*sizeof(float));
    float* B = (float*)malloc(nb*mb*sizeof(float));
    float* C = (float*)malloc(na*mb*sizeof(float));

    for(int i=0;i<na;i++)
    {
        for(int j=0;j<ma;j++)
        {
            A[i*ma+j] = i+j;
        }
    }

    for(int i=0;i<nb;i++)
    {
        for(int j=0;j<mb;j++)
        {
            B[i*mb+j] = i+j;
        }
    }

    matmul(A,B,C,na,ma,nb,mb,atoi(argv[5]));

    // for(int i=0;i<na;i++)
    // {
    //     for(int j=0;j<ma;j++)
    //     {
    //         std::cout<<A[i*ma+j]<<' ';
    //     }
    //     std::cout<<'\n';
    // }
    // std::cout<<'\n';

    // for(int i=0;i<nb;i++)
    // {
    //     for(int j=0;j<mb;j++)
    //     {
    //         std::cout<<B[i*mb+j]<<' ';
    //     }
    //     std::cout<<'\n';
    // }
    // std::cout<<'\n';

    // for(int i=0;i<na;i++)
    // {
    //     for(int j=0;j<mb;j++)
    //     {
    //         std::cout<<C[i*mb+j]<<' ';
    //     }
    //     std::cout<<'\n';
    // }
    // std::cout<<'\n';

    free(A);
    free(B);
    free(C);
}