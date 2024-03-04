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
    
}