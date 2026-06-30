#include<iostream>

void matmul(float* A,float* B,float* C,int na,int ma,int nb,int mb)
{
    if(ma!=mb)
    {
        std::cout<<"not possible\n";
        return;
    }

    float avg_time=0;

    for(int l=0;l<5;l++)
    {
        cudaEvent_t start;
        cudaEvent_t stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        for(int i=0;i<na;i++)
        {
            for(int j=0;j<mb;j++)
            {
                C[i*mb+j]=0;
                for(int k=0;k<ma;k++)
                {
                    C[i*mb+j] += A[i*ma+k]*B[k*mb + j];
                }
            }
        }

        cudaEventRecord(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed,start,stop);

        avg_time+=elapsed;
    }

    std::cout<<"elapsed time: "<<avg_time/5<<"ms\n";
}

int main()
{
    int ma=1024;
    int nb=1024;
    int mb=1024;
    int na=1024;
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