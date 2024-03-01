__global__ void launchTailGraph(cudaGraphExec_t graph){
    cudaGraphLaunch(graph,cudaStreamGraphTailLaunch);
}
void graphSetup(){
    cudaGraphExec_t gExec1,gExec2;
    cudaGraph_t g1,g2;
    // create instantiate and upload the device graph 
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2,g2,cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2,stream);

    // create and instantiate the launching graph 
    cudaStreamBeginCapture(stream,cudaStreamCaptureModelGlobal);
    launchTailGraph<<<1,1,0,stream>>>(gExec2);
    cudaStreamEndCapture(stream,&g1);
    cudaGraphInstantiate(&gExec1,g1);
    // launch the host graph which will in turn launch the device graph 
    cudaGraphLaunch(gExec1,stream);
    
}