__global__ void transformerKernel(float* output,cudaTextureObject_t texObj,int width,int height,float theta){
    //calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 

    float u = x/(float)width;
    float v = y/(float)height;

    // transform coordinates 
    u -= 0.5f;
    v -= 0.5f;
    float tu = u*cosf(theta) - v*sinf(theta)+0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // Read from texture and write to global memory
    output[y * width + x] = tex2D<float>(texObj, tu, tv);

}

//host code 
int main(){
    const int height = 1024;
    const int width = 1024;
    float angle = 0.5;
    // allocate and set some host data 
    float *h_data = (float *)std::malloc(sizeof(float)*width*height);
    for(int i=0;i<height*width;++i)
        h_data[i] = i;
    
    //allocate cuda array in device memory 
    cudaChannelFormatDesc channelDesc = cudacreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    //设置源的间距（以所指向的2D阵列的字节为单位的内存宽度
    //到src，包括填充），我们没有任何填充
    const size_t spitch = width *sizeof(float);
    //copy data located at address h_data in host memory to device memory 
    cudaMemcpy2DToArray(cuArray,0,0,h_data,spitch,width*sizeof(float),height,cudaMemcpyHostToDevice);

    //specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc,0,sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    //specify texture object parameters 
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // create texture object 
    cudaTextureObject_t texObj =0 ;
    cudaCreateTextureObject(&texObj,&resDesc,&texDesc,NULL);
    
    //allocate result of transformers in device memory 
    float* output;
    cudaMalloc(&output,width*height*sizeof(float));

    //invoke kernel 
    dim3 threadsperBlock(16,16);
    dim3 numBlocks((width+threadsperBlock.x-1)/threadsperBlock.x,(height+threadsperBlock.y-1)/threadsperBlock.y);
    transformerKernel<<<numBlocks,threadsperBlock>>>(output,textObj,width,height,angle);

    //copy data from device back to host 
    cudaMemcpy(h_data,output,width*height*sizeof(float),cudaMemcpyDeviceToHost);

    //destroy texture object 
    cudaDestroyTextureObject(texObj);

    //free device memory 
    cudaFreeArray(cuArray);
    cudaFree(output);

    //free host memory 
    free(h_dat);

    return 0;

}