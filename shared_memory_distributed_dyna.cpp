
//通过可扩展启动启动
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = array_size / threads_per_block;
    config.gridDim = threads_per_block;
    // cluster size depends on the historgram size 
    // (cluster_size==1) implies no distributed shared memory,  just thread block local shared memory
    int cluster_size = 2; // size  2 is an example here 
    int nbins_per_block = nbins/cluster_size;

    //dynamic sshared memory size is per block 
    // distributed shared mmeory size = cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = nbins_per_block*sizeof(int);

    CUDA_CHECK(::cudaFuncSetAttribute((void*)clusterHist_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,config.cudaFuncAttributeMaxDynamicSharedMemorySize));

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attributep[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.numAttrs = 1;
    config.attrs = attribute;
    cudaLaunchKernelEx(&config,clusterHist_kernel,bins,nbins,nbins_per_block,input,array_size);


    }