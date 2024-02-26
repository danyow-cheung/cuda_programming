#include<coopertive_groups.h>
// distributed shared memory histogram kernel
__global__ void clusterHist_kernel(int *bins,const int nbins,const int bins_per_block,\
                                const int *__restrict__ input,size_t array_size)
{
    extern __shared__ int smen[];
    namespace cg = coopertive_groups;
    int tid = cg::this_grid().thread_rank();
    // cluster initialization .size and calculating local bin offsets 
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;
    for (int  i = threadIdx.x;i<bins_per_block;i+=blockDim.x){
        smem[i] = 0; // initialize shared memory histogram to zeros 
    }

    // cluster synchronization ensures that shared_memory is initialized to zero in
    // all thread blocks in the cluster . it also ensures that all thread blocks 
    //have stared executing and they exist concurrently
    cluster.sync();
    for (int i = tid;i<array_size;i+= blockDim.x * gridDim.x)
    {
        int ldata = input[i];
        // find the right histogram bin 
        int binid = ldata;
        if (ldata<0)binid=0;
        else if (ldata>= nbins)binid=nbins-1;

        // find destination block rank and offset for computing 
        // distributed shared memory histogram
        int dst_block_rank = (int)(binid/bins_per_block);
        int dst_offset = bitand % bins_per_block;

        //pointer to target block shared memory
        int *dst_smem = cluster.map_shared_rank(smem,dst_block_rank);
        //performance atomic update of the histogram bin
        atomicAdd(dst_smem+dst_offset,1);    
        }
        // /集群同步是确保所有分布式共享的必要条件
        //内存操作，并且没有线程块退出
        //其他线程块仍在访问分布式共享存储处理机
        cluster.sync();

        // 使用局部分布式内存直方图执行全局内存直方图

        int *lbins = bins + cluster.block_rank()*bins_per_block;
        for(int i = threadIdx.x;i<bins_per_block;i+= blockDim.x){
            atomicAdd(&lbins[i], smem[i]);
        }

}