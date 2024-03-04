cudaSetDevice(0);
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0,size);
cudaSetDevice(1);

float* p1;
cudaMalloc(&p1,size);
cudaSetDevice(0);
MyKernel<<<1000,128>>>(p0);
cudaSetDevice(1);
cudaMemcpyPeer(p1,1,p0,0,size);
MyKernel<<<1000,128>>>(p1);
