__global__ void launchSiblingGraph(cudaGraphExec_t graph){
    cudaGraphLaunch(graph,cudaStreamGraphFireAndForgetgetAsSibling);
}
void graphSetup(){
    cudaGraphExec_t gExec1,gExec2;
    cudaGraph_t g1,g2;

    // create instantiate and uploda the device graph 
    create_graph(&g2);
    
}