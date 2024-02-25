// kerenl defintion 
__global__ void VecAdd(float *A,float *B,float *C,int N){
    int i = blockDim.x  * blockIdx.x + threadsIdx.x;
    if (i<N){
        C[i] = A[i]+B[i];
    }

}

// host code 

int main(){
    int N = ...;
    size_t  size = N * sizeof(float);
    // allocate input vectors h_A and h_B in host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // inititliaze input vectors 


    // allocate input vectors
    float* d_A;
    cudaMalloc(&d_A,size);
    float* d_B;
    cudaMalloc(&d_B,size);
    float* d_C;
    cudaMalloc(&d_C,size);

    // copy vectors from host memory to device memory 
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    // invoke kernel 
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

    VecAdd<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);
    
    // copy result from device memory to host memory 
    //h_C contains the result in host memory 
    cudaMemcpy(h_C,d_C,size,cudaMemcpDeviceToHost);
    // free device memory 
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // free host mmemory 
}
