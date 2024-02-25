//以下代码示例说明了通过运行时 API 访问全局变量的各种方法：
__constant__ float constData[256];
float data[256];
cudaMemcpyToSymbol(constData,data,sizeof(data));
cudaMemcpyToSymbol(data,constData,sizeof(data));

__device__ float devData;
float value = 3.14f;
cudaMemcpyToSymbol(devData,&value,sizeof(float));

__device__ float* devPointer;
float* ptr;
cudaMalloc(&ptr,256*sizeof(float));
cudaMemcpyToSymbol(devPointer,&ptr,sizeof(ptr));