// --------3D数组循环设备遍历
// Host code 
int width = 64 ,height = 64,depth = 64;
cudaExtent  extent = make_cudaExtent(width *sizeof(float),height,width);
cudaPitchedPtr devicePitchedPtr;
cudaMalloc3D(&devicePitchedPtr,extent);
MyKernel<<<100,512>>(devPtr,width,height,depth);

// device code 
__global__ void MyKernel(cudaPitchedPtr,devicePitchedPtr,int width,int height,int depth){
    char* devPtr = devicePitchedPtr.ptr;
    size_t pitch = devicePitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0;z<depth;++z){
        char* slice = devPtr +z*slicePitch;
        for(int y =0;y<height;++y){
            float* row = (float*)(slice + y*pitch);
            for(int x =0;x<width;++x){
                float element = row[x];
            }
        }
    }

}

