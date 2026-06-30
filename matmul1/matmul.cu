#include<iostream>

#define cf 4

__global__ void matrix_multi_shared(float* A,float* B,float* C,int na,int ma,int nb,int mb,int tile_dim)
{
    extern __shared__ float sh_mem[];

    float* sh_A = sh_mem;
    float* sh_B = sh_mem + tile_dim*tile_dim;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float oc[cf*cf];
    for(int p=0;p<cf*cf;p++) oc[p]=0.0;

    for(int m=0;m<(ma+tile_dim-1)/tile_dim;m++)
    {

        for(int tiy = ty;tiy<tile_dim;tiy+=blockDim.y)
        {
            for(int tix = tx;tix<tile_dim;tix+=blockDim.x)                
            {
                int i = tile_dim*blockIdx.y + tiy;
                int j = tile_dim*blockIdx.x + tix;
                
                if(i<na && (m*tile_dim+tix)<ma)
                {
                    sh_A[tiy*tile_dim + tix] = A[i*ma + m*tile_dim + tix];
                }
                else sh_A[tiy*tile_dim + tix] = 0.0;

                if(j<mb && (m*tile_dim+tiy)<nb) 
                {
                    sh_B[tiy*tile_dim + tix] = B[(m*tile_dim+tiy)*mb + j];
                }
                else sh_B[tiy*tile_dim + tix] = 0.0;                    
            }
        }

        __syncthreads();

        int p=0;
        for(int tiy = ty;tiy<tile_dim;tiy+=blockDim.y)
        {
            for(int tix = tx;tix<tile_dim;tix+=blockDim.x)                
            {     
                for(int k=0;k<tile_dim;k++) oc[p] += sh_A[tiy*tile_dim + k]*sh_B[k*tile_dim + tix];
                p++;
            }
        }

        __syncthreads();

    }

    int p=0;
    for(int tiy = ty;tiy<tile_dim;tiy+=blockDim.y)
    {
        for(int tix = tx;tix<tile_dim;tix+=blockDim.x)                
        {
            int i = tile_dim*blockIdx.y + tiy;
            int j = tile_dim*blockIdx.x + tix;

            if(i<na && j<mb) 
            {
                C[i*mb + j] = oc[p];
            }

            p++;
        }
    }

}

void matmul_shared(float* A,float* B,float* C,int na,int ma,int nb,int mb,int block_dim)
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

    int tile_dim = cf * block_dim;

    dim3 threads(block_dim,block_dim);
    dim3 blocks((mc+tile_dim-1)/tile_dim,(nc+tile_dim-1)/tile_dim);

    int shared_memory = 2*tile_dim*tile_dim*sizeof(float);
    matrix_multi_shared<<<blocks,threads,shared_memory>>>(dA,dB,dC,na,ma,nb,mb,tile_dim);
    cudaDeviceSynchronize();

    cudaMemcpy(C,dC,sizeof(float)*nc*mc,cudaMemcpyDefault);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char* argv[]) // na ma nb mb block_dim 
{
    int na = atoi(argv[1]);
    int ma = atoi(argv[2]);
    int nb = atoi(argv[3]);
    int mb = atoi(argv[4]);
    float* A=NULL;
    float* B=NULL;

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