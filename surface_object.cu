// simple copy kernel 
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                            cudaSurfaceObject_t outputSurObj,
                            int width,int height)
{
    //calculate surface coordinations 
    unsigned int x=  blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y=  blockIdx.y * blockDim.y + threadIdx.y;
    if (x<width&&y<height){
        uchar4 data;
        //read from input surface
        surf2Dread(&data,inputSurfObj,x*4,y);
        //write to output surface 
        surf2DWrite(data,outputSurfObj,x*4,y);
    }
}

// host code 
int main(){
    const int height = 1024;
    const int width = 1024;
    //allocate and set some host data 
    unsigned char *h_data = (unsigned char *)std::malloc(sizeof(unsigned char)*width*height*4);
    for(int i=0;i<height*width*4;++i){
        h_data[i] = i;
    }

    //allocate cuda arrays  in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,8,8,8,cudaChannelFormatKindUnsigned);
    cudaArray_t cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,cudaArraySurfaceLoadStore);

    cudaArray_t cuOutputArray;
    cudaMallocArray(&cuOutputArray,&channelDesc,width,height,cudaArraySurfaceLoadStore);

    //set pitch of the source (the width in memory in bytes of the 2d array )
    // pointed to by src,including padding ,we don't have any padding 
    const size_t spitch = 4*width*sizeof(unsigned char);
    //copy data located at address h_data in host memory to device memory 
    cudaMemcpy2DTOArray(cuInputArray,0,0,h_data,spitch,4*width*sizeof(unsigned char),height,cudaMemcpyHostToDevice);

    //specify surface 
    struct cudaResourceDesc resDesc;
    memst(&resDesc,0,sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    //create the surface objects
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurObj,&resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject(&outputSurfObj,&resDesc);

    //invoke kernel 
    dim3 threadsperBlock(16,16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                    (height + threadsperBlock.y - 1) / threadsperBlock.y);
    copyKernel<<<numBlocks, threadsperBlock>>>(inputSurfObj, outputSurfObj, width,
                                                height);
    //copy data from device back to host 
    cudaMemecpy2DFromArray(h_data,spitch,cuOutputArray,0,0,4*width*sizeof(unsigned char),height,cudaMemcpyDeviceToHost);

    //destropy surface objects 
    cudaDestroySurfaceObject(inputSurfObj);
    cudaDestroySurfaceObject(outputSurObj);

    //free device memory 
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);
    //free host memory
    free(h_data);
    return 0 ;
}