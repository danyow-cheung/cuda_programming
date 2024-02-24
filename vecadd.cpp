// kerenl defintion 
__global__ void MatAdd(float A[N][N],float B[N][N],float c[N][N]){
    int i = block.Idx.x * blockDim.x + threadsIdx.x;
    int j = block.Idx.y * blockDim.y + threadsIdx.y;
    if (i<N && j<N)
    {
        c[i][j] = a[i][j] + b[i][j];
    }
}

int main(){
  // kernel invocation 
  dim3 threadsPerBlock(16,16);
  dim3 numBlocks(N/threadsPerBlock.x , N/threadsPerBlock.y);
  MatAdd<<numBlocks,threadsPerBlock>>(A,B,C);
}
