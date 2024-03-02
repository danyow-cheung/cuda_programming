#include <stdio.h>
#include <cuda_runtime.h>

int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}

// 不太懂，按照的都是官网源码但是报错 
// show_devices.cu
// ./show_devices.cu(5): error: this declaration has no storage class or type specifier

// ./show_devices.cu(5): error: declaration is incompatible with "cudaError_t cudaGetDeviceCount(int *)"
// C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\cuda_device_runtime_api.h(132): here

// ./show_devices.cu(5): error: a value of type "int *" cannot be used to initialize an entity of type "int"

// ./show_devices.cu(7): error: expected a declaration

// At end of source: warning: parsing restarts here after previous syntax error

// 4 errors detected in the compilation of "./show_devices.cu".
