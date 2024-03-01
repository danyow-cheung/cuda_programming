__device__ int relaunchcCount = 0;
__global__ void relaunchSelf(){
    int relaunchMax = 100;
    if (threadIdx.x==0){
        if (relaunchcCount<relaunchMax){
            cudaGraphLaunch(cudaGetCurrentGraphExec(),cudaStreamGraphTailLaunch);
        }
        relaunchCount ++;
    }
}