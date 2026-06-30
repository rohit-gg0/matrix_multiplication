#include<iostream>

#define cf 2

__global__ void matrix_multi_shared(float* A, float* B, float* C,int na,int ma,int nb,int mb,int tile_size)
{
    extern __shared__ float sh_mem[];
    float* sh_A = sh_mem;
    float* sh_B = sh_mem + tile_size*tile_size;   
    
    float sum[cf*cf];
    for(int i=0;i<cf*cf;i++) sum[i]=0.0f;

    for(int m=0;m<(ma+tile_size-1)/tile_size;m++)
    {
        for(int tiy=threadIdx.y ;tiy<tile_size;tiy+=tile_size/cf)
        {
            for(int tix=threadIdx.x ;tix<tile_size;tix+=tile_size/cf)
            {
                int i = tile_size*blockIdx.y + tiy;
                int j = tile_size*blockIdx.x + tix;
                
                if(i<na && (m*tile_size+tix)<ma)
                {
                    sh_A[tiy*tile_size + tix] = A[i*ma + m*tile_size + tix];
                }
                else sh_A[tiy*tile_size + tix] = 0.0;

                if(j<mb && (m*tile_size+tiy)<nb) 
                {
                    sh_B[tiy*tile_size + tix] = B[(m*tile_size+tiy)*mb + j];
                }
                else sh_B[tiy*tile_size + tix] = 0.0;  
            }
        }

        __syncthreads();

        int p=0;
        for(int tiy=threadIdx.y ;tiy<tile_size;tiy+=tile_size/cf)
        {
            for(int tix=threadIdx.x ;tix<tile_size;tix+=tile_size/cf)                
            {     
                for(int k=0;k<tile_size;k++) sum[p] += sh_A[tiy*tile_size + k]*sh_B[k*tile_size + tix];
                p++;
            }
        }

        __syncthreads();
    }

    int p=0;
    for(int tiy=threadIdx.y ;tiy<tile_size;tiy+=tile_size/cf)
    {
        for(int tix=threadIdx.x ;tix<tile_size;tix+=tile_size/cf)                
        {
            int i = tile_size*blockIdx.y + tiy;
            int j = tile_size*blockIdx.x + tix;

            if(i<na && j<mb) 
            {
                C[i*mb + j] = sum[p];
            }

            p++;
        }
    }
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

    int tile_size = cf*block_size;

    dim3 threads(block_size,block_size);
    dim3 blocks((mc+tile_size-1)/tile_size,(nc+tile_size-1)/tile_size);

    int shared_memory = 2*tile_size*tile_size*sizeof(float);

    matrix_multi_shared<<<blocks,threads,shared_memory>>>(dA,dB,dC,na,ma,nb,mb,tile_size);
    cudaDeviceSynchronize();

    cudaMemcpy(C,dC,sizeof(float)*nc*mc,cudaMemcpyDefault);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
} 


int main(int argc, char* argv[]) // na ma nb mb block_size
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

    for(int i=0;i<na;i++)
    {
        for(int j=0;j<ma;j++)
        {
            std::cout<<A[i*ma+j]<<' ';
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';

    for(int i=0;i<nb;i++)
    {
        for(int j=0;j<mb;j++)
        {
            std::cout<<B[i*mb+j]<<' ';
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';

    for(int i=0;i<nc;i++)
    {
        for(int j=0;j<mc;j++)
        {
            std::cout<<C[i*mc+j]<<' ';
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
}