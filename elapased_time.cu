cudaEventRecord(start,0);
for(int i =0;i<2;++i){
    cudaMemcpyAsync(inputDev+i*size ,inputHost+i*size ,size , cudaMemcpyHostToDevice,stream[i] );
    MyKernel<<<100,512,0,,stream[i]>>>(outputDev+i*size,inputDev+i*size,size);
    cudaMemcpyAsync(outputHost+i*size ,outputDev+i*size ,size , cudaMemcpyHostToDevice,stream[i]);
}
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
float elapsed_time;
cudaEventElapsedTime(&elapsed_time,start,stop);