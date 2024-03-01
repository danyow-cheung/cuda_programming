// kerenel definition
// compile time cluster size 2 in x-dimension and 1 in Y and z dimension
__global__ void __cluster_dims__(2,1,1) cluster_kernel(float *input,float *output)
{

}

int main(){
    float *input ,*output;
    // kernel invocation with compile time cluster size 
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(N/threadsIdx.x,N/threadsIdx.y);
    // the grid dimension is not affected by cluster launch ,and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks,threadsPerBlock>>>(input,output);
}


