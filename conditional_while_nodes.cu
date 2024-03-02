__global__ void loopKernel(cudaGraphConditionalHandle handle){
    static int count = 10;
    cudaGraphSetConditional(handle,--count?1:0);
}

void graphSetup(){
    cudaGrapht_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t node;
    void *kernelArgs[1];

    cuGraphCreate(&graph,0);
    cudaGraphConditionalHandle  handle;
    cudaGraphConditionalHandleCreate(&handle,graph,1,cudaGraphCondAssignDefault);

    cudaGraphNodeParams cParams = {cudaGraphNodeTypeConditional};
    cParams.conditional.handle = handle;
    cParams.conditional.type = cudaGraphCondTypeWhile;
    cParams.conditional.size =1 ;
    cudaGraphAddNode(&node,graph,NULL,0,&cParams);

    cudaGrapht_t bodyGraph = cParams.conditional.phGraph_out[0];
    cudaGraphNodeParams params = {cudaGraphNodeTypeKernel};
    params.kernel.func =  (void *)loopKernel;
    params.kernel.gridDim.x = params.kernel.gridDim.y = params.kernel.gridDim.z =1;
    params.kernel.blockDim.x = params.kernel.blockDim.y = params.kernel.blockDim.z =1;
    params.kernel.kernelParams = kernelArgs;
    kernelArgs[0] = &handle;

    cudaGraphAddNode(&node,bodyGraph,NULL,0,&params);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
}