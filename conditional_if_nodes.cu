__global__ void setHandle(cudaGraphConditionalHandle handle){
    ...
    cudaGraphSetConditional(handle,value);
    ...
}

void graphSetup(){
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];
    int value = 1;

    cudaGraphCreate(&graph,0);
    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle,graph);
    
    //use a kernel upstream of the conditional to set the handle value 
    cudaGraphNodeParams params = {cudaGraphNodeTypeKernel};
    params.kernel.func = (void)*setHandle;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z = 1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z = 1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;
    cudaGraphNodeAddNode(&node,graph,null,0,&params);

    cudaGraphNodeParams cParams = {cudaGraphNodeTypeConditional};
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeIf;
    cParams.conditional.size = 1;
    cudaGraphAddNode(&node,graph,&node,1,&cParams);

    cudaGraph_t bodyGraph = cParams.conditional.phGraph_out[0];
    // populate the body of the conditional node 
    cudaGraphAddNode(&node,bodyGraph,NULL,0,&params);

    cudaGraphInstantiate(&graphExec,graph,NULL,NULL,0);
    cudaGraphLaunch(graphExec,0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestory(graph);
}