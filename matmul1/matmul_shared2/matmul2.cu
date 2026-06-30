#include<iostream>

#define tile_factor 2

__global__ void matrix_multi_shared(float* A, float* B, float* C,int na,int ma,int nb,int mb,int tile_size)
{
    extern __shared__ float sh_mem[];
    float* sh_A = sh_mem;
    float* sh_B = sh_mem + tile_size*tile_size;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int ty2 = ty + blockDim.y;
    int tx2 = tx + blockDim.x;

    int i = blockDim.y*blockIdx.y + ty;
    int j = blockDim.x*blockIdx.x + tx;

    int i2 = blockDim.y*blockIdx.y + ty2;
    int j2 = blockDim.x*blockIdx.x + tx2;

    float oc=0;

    for(int m=0;m<(ma+tile_size-1)/tile_size;m++)
    {
        if(i<na && (m*tile_size + tx)<ma) sh_A[ty*tile_size+tx] = A[i*ma + (m*tile_size + tx)];
        else sh_A[ty*tile_size+tx] = 0.0;

        if(j<mb && (m*tile_size+ty)<nb) sh_B[ty*tile_size+tx] = B[(m*tile_size+ty)*mb + j];
        else sh_B[ty*tile_size+tx] = 0.0;

        if(i2<na && (m*tile_size + tx2)<ma) sh_A[ty2*tile_size+tx2] = A[i2*ma + (m*tile_size + tx2)];
        else sh_A[ty2*tile_size+tx2] = 0.0;

        if(j2<mb && (m*tile_size+ty2)<nb) sh_B[ty2*tile_size+tx2] = B[(m*tile_size+ty2)*mb + j2];
        else sh_B[ty2*tile_size+tx2] = 0.0;

        if(i2<na && (m*tile_size + tx)<ma) sh_A[ty2*tile_size+tx] = A[i2*ma + (m*tile_size + tx)];
        else sh_A[ty2*tile_size+tx] = 0.0;

        if(j<mb && (m*tile_size+ty2)<nb) sh_B[ty2*tile_size+tx] = B[(m*tile_size+ty2)*mb + j];
        else sh_B[ty2*tile_size+tx] = 0.0;

        if(i<na && (m*tile_size + tx2)<ma) sh_A[ty*tile_size+tx2] = A[i*ma + (m*tile_size + tx2)];
        else sh_A[ty*tile_size+tx2] = 0.0;

        if(j2<mb && (m*tile_size+ty)<nb) sh_B[ty*tile_size+tx2] = B[(m*tile_size+ty)*mb + j2];
        else sh_B[ty*tile_size+tx2] = 0.0;

        __syncthreads();

        for(int k=0;k<tile_size;k++) oc+= sh_A[ty*tile_size+k]*sh_B[k*tile_size+tx];

        __syncthreads();
    }

    if(i<na && j<mb) C[i*mb+j] = oc;

}

void matmul_shared(float* A,float* B,float* C,int na,int ma,int nb,int mb,int block_size)
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

    int tile_size = tile_factor*block_size;

    dim3 threads(block_size,block_size);
    dim3 blocks((mc+threads.x-1)/threads.x,(nc+threads.y-1)/threads.y);

    int shared_memory = 2*tile_size*tile_size*sizeof(float);

    matrix_multi_shared<<<blocks,threads,shared_memory>>>(dA,dB,dC,na,ma,nb,mb,tile_size);
    cudaDeviceSynchronize();

    cudaMemcpy(C,dC,sizeof(float)*nc*mc,cudaMemcpyDefault);

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
    float* A=NULL;
    float* B=NULL;

    // std::cin>>na>>ma>>nb>>mb;

    cudaMallocHost(&A,sizeof(float)*na*ma);
    cudaMallocHost(&B,sizeof(float)*nb*mb);

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

    float* C=NULL;
    int nc=na;
    int mc=mb;

    cudaMallocHost(&C,sizeof(float)*nc*mc);

    matmul_shared(A,B,C,na,ma,nb,mb,atoi(argv[5]));

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