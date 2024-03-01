__global__ void launchFiredAndForgetGraph(cudaGraphExec_t graph){
    cudaGraphLaunch(graph,cudaStreamGraphFiredAndForget);
}

void graphSetup(){
    cudaGraphExec_t gExec1,gExec2;
    cudaGraph_t g1,g2;

    // create .instanitate and upload the device graph 
    create_graph(&g2);
    cudaGraphInstantiate(&gExec2,g2,cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2,stream);

    // create and instantitate the launching graph 
    cudaStreamBeginCapture(stream,cudaStreamCaptureModelGlobal);
    launchFiredAndForgetGraph<<<1,1,0,stream>>>(gExec2);
    cudaStreamEndCapture(stream,&g1);
    cudaGraphInstantiate(&gExec1,g1);
    // launch the host origin which will in turn launch the device graph 
    cudaGraphLaunch(gExec1,stream);
}