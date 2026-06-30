#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ __cluster_dims__(2,1,1) void dsm_test(int *out)
{
    extern __shared__ int smem[];

    cg::cluster_group cluster = cg::this_cluster();

    int rank = cluster.block_rank();

    smem[threadIdx.x] = rank;

    cluster.sync();

    int *remote = cluster.map_shared_rank(smem, 1);

    out[rank * blockDim.x + threadIdx.x] = remote[threadIdx.x];
}

int main()
{
    int threads = 128;

    int blocks = 2;

    int* dout;
    cudaMalloc(&dout,blocks*threads*sizeof(int));

    dsm_test<<<blocks,threads,threads*sizeof(int)>>>(dout);

    int* out = (int*)malloc(threads*blocks*sizeof(int));

    cudaMemcpy(out,dout,threads*blocks*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<threads*blocks;i++) 
    {
        printf("%d ",out[i]);
    }
    printf("\n");
}