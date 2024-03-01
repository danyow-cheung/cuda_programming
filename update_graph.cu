cudaGraphExec_t graphExec = NULL;
for (int i =0 ;i<10;i++){
    cudaGraph_t graph;
    cudaGraphExecUpdateResult updateResult;
    cudaGraphNode_t errNode;

    // in this example we use stream capture to create the graph 
    // you can also use the graph api to produce a graph 
    cudaStreamBeginCapture(stream,cudaStreamCaptureModelGlobal);
    // call a user-defined stream based workload for example 
    do_cuda_work(stream);

    cudaStreamEndCapture(stream,&graph);
    // if we've already instantiated the graph,try to update 
    // it directly and avoid the instantiation overhead 
    if (graphExec!=NULL){
        cudaGraphExecUpdate(graphExec,graph,&errorNode,&updateResult);
    }
    // Instantiate during the first iteration or whenever the update
    // fails for any reason
    if (graphExec==NULL || updateResult != cudaGraphExecUpdateSuccess){
        if (graphExec!=NULL){
            cudaGraphExecDestory(graphExec);
        }

        // instantiate graphExec from graph
        cudaGraphInstantiate(&graphExec,graph,NULL,NULL,0);
    }

    cudaGraphDestory(graph);
    cudaGraphLaunch(graphExec,stream);
    cudaStreamSynchronize(stream);
}
