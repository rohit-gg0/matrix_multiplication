#include<iostream>

#define block_size 16

__global__ void matrix_multi_global(float* A,float* B,float* C,int na,int ma,int nb,int mb)
{
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    if(row<na && col<mb)
    {
        float oc=0;
        for(int k=0;k<ma;k++)
        {
            oc += A[row*ma + k]*B[k*mb + col];
        }
        C[row*mb + col] = oc;
    }
}

void matmul(float* A,float* B,float* C,int na,int ma,int nb,int mb)
{
    if(ma!=nb)
    {
        std::cout<<"Not Possible\n";
        return;
    }
    
    float* dA=NULL;
    float* dB=NULL;
    float* dC=NULL;

    int nc=na;
    int mc=mb;

    cudaMalloc(&dA,sizeof(float)*na*ma);
    cudaMalloc(&dB,sizeof(float)*nb*mb);
    cudaMalloc(&dC,sizeof(float)*nc*mc);

    cudaMemcpy(dA,A,sizeof(float)*na*ma,cudaMemcpyDefault);
    cudaMemcpy(dB,B,sizeof(float)*nb*mb,cudaMemcpyDefault);


    dim3 threads(block_size,block_size);
    dim3 blocks((mb+threads.x-1)/threads.x,(na+threads.y-1)/threads.y);

    float avg_time = 0;

    for(int i=0;i<1000;i++)
    { 
        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        matrix_multi_global<<<blocks,threads>>>(dA,dB,dC,na,ma,nb,mb);

        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float elapsed;
        cudaEventElapsedTime(&elapsed,start,stop);

        avg_time +=elapsed;
    }

    std::cout<<"elapsed time: "<<avg_time/1000<<"ms\n";

    cudaMemcpy(C,dC,sizeof(float)*nc*mc,cudaMemcpyDefault);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main()
{
    int na=1024;
    int ma=1024;
    int nb=1024;
    int mb=1024;
    float* A=NULL;
    float* B=NULL;

    cudaMallocHost(&A,sizeof(float)*na*ma);
    cudaMallocHost(&B,sizeof(float)*nb*mb);

    for(int i=0;i<na;i++)
    {
        for(int j=0;j<ma;j++)
        {
            A[i*ma+j] = 1;
        }
    }

    for(int i=0;i<nb;i++)
    {
        for(int j=0;j<mb;j++)
        {
            B[i*mb+j] = -1;
        }
    }

    float* C=NULL;
    int nc=na;
    int mc=mb;

    cudaMallocHost(&C,sizeof(float)*nc*mc);

    matmul(A,B,C,na,ma,nb,mb);

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

    // for(int i=0;i<nc;i++)
    // {
    //     for(int j=0;j<mc;j++)
    //     {
    //         std::cout<<C[i*mc+j]<<' ';
    //     }
    //     std::cout<<'\n';
    // }
    // std::cout<<'\n';

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
}