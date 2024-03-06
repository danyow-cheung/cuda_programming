ID3D10Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D10Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int maint(){
    int dev;
    //get a cuda-enabled adapter 
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for(unsigned int i=0;!adapter;++i){
        if (FAILED(factory->EnumAdapters(i,&adapter)))
            break;
        if (cudaD3D10GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();
    // create swap chain and device 
    ...
    D3D10CreateDeviceAndSwapChain(adapter,
        
        D3D10_DRIVER_TYPE_HARDWARE, 0,
        D3D10_CREATE_DEVICE_DEBUG,
        D3D10_SDK_VERSION,
        &swapChainDesc, &swapChain,
        &device);
    adapter->Release();
    // use same devic e
    cudaSetDevice(dev);

    //create vertex buffer adn register it with cuda 
    unsigned int size = width  *height *sizeof(CUSTOMVERTEX);

    D3D10_BUFFER_DESC bufferDesc;
    bufferDesc.Usage          = D3D10_USAGE_DEFAULT;
    bufferDesc.ByteWidth      = size;
    bufferDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags      = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D10RegisterResource(&positionsVB_CUDA,
                                      positionsVB,
                                      cudaGraphicsRegisterFlagsNone);
                                      cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                      cudaGraphicsMapFlagsWriteDiscard);

    // Launch rendering loop
    while (...) {
        ...
        Render();
        ...
    }
    ...
}


void Render(){
    //map vertext buffer for writting from cuda 
    float4* positions;
    cudaGraphicsMapResource(1,&positionsVB_CUDA,0);
    size_t num_bytes;

    cudaGraphicsResourceGetMappedPointer((void**)&positions,&num_bytes,positionsVB_CUDA);


    //executre kernel 
    dim3 dimBlock(16,1,6,1);
    dim3 dimGrid(width/dimBlock.x,height/dimBlock.y,1);
    createVertices<<dimGrid,dimBlock>>>(positions,time,width,height);

    //unmap vertext buffer 
    cudaGraphicsUnmapResources(1,&positionsVB_CUDA,0);

    //draw and present 
    ...
}

void releaseVB(){
    cudaGraphicsUnregisterResources(positionsVB_CUDA);
    positionsVB->Release();
}

__global__ void createVertices(float4* positions,float time,unsigned int width,unsigned int height){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate uv coordinates
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // Calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // Write positions
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
                
}