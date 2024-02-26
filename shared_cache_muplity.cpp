// matrices are stored in row-major order 
//M(row,col) =* (M.elements + row*M.stride+col)

typedef struct  
{
    int width;
    int height;
    int stride;
    float* elements;
}Matrix;

// get a matrix elements
// note:https://blog.csdn.net/zdlnlhmj/article/details/104896470
__device__ float GetElement(const Matrix A,int row,int col){
    return A.elements[row*A.stride+col];
}

// set a matrix elements 
__device__ void SetElement(Matrix A,int row,int col,float value){
    A.elements[row*A.stride+col] = value;
}
// get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is 
// Located COL sub-matrices to the right and row sub-matrices down form the 
// upper-left conrner of A
__device__ Matrix GetSubMatrix(Matrix A,int row,int col){
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride*BLOCK_SIZE*row+BLOCK_SIZE*col];
    return Asub;
}

// thread block size 
#define BLOCK_SIZE 16
__global__ void MatMulKernel(const Matrix ,const Matrix,Matrix);

// matrix multiplication - host code 
// matrix dimensions are assumend to be multiples of BLOCK_SIZE 
void MatMul(const Matrix A,const Matrix B,Matrix C){
    // load A and B to device memory 
    Matrix d_A;
    d_A.width = d_A.stride = A.width ;d_A.height = A.height;

    size_t size = A.width *A.height *sizeof(float);
    cudaMalloc(&d_A.elements,size);
    cudaMemcpy(d_A.elements,A.elements,size,cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width ;d_B.height = B.height;
    size = B.width *B.height *sizeof(float);
    cudaMalloc(&d_B.elements,size);
    cudaMemcpy(d_B.elements,B.elements,size,cudaMemcpyHostToDevice);

    // allocate c in drive memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width ;d_C.height = C.height;
    size = C.width *C.height *sizeof(float);
    cudaMalloc(&d_C.elements,size);
    // invoke kernel 
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(B.width/dimBlock.x,A.height/dimBlock.y);
    MatMulKernel<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);
    // read c from device memory 
    cudaMemcpy(C.elements,d_C.elements,size,cudaMemcpyDeviceToHost);
    // free device memory 
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A,Matrix B,Matrix C){
    // block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // each thread block computes one sub-matrix Csub of C 
    Matrix Csub = GetSubMatrix(C,blockRow,blockCol);

    //each thread computes one element of Csub 
    // by accumulating results into Cvalue 
    float cvalue = 0;
    // thread row and column within Csub 
    int row = threadIdx.y;
    int col = threadIdx.x;

    // loop over all the sub-matrices of A and B that are 
    // required to compute Csub 
    // Multiply each pair of sub-matrices together 
    // and accumulate the results 
    for (int m =0;m<(A.width/BLOCK_SIZE);++m){
        // get sub-matrix Asub of A 
        Matrix Asub = GetSubMatrix(A,blockRow,m);
        Matrx Bsub = GetSubMatrix(B,m,blockCol);
        // shared memory used to store Asub and Bsub respecitively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // load asub and bsub from device memory to shared memory 
        // each thread loads one element of each sub-matrix 
        As[row][col] = GetElement(Asub,row,col);
        Bs[row][col] = GetElement(Bsub,row,col);

        //同步以确保加载了子矩阵
        //在开始计算之前
        __syncthreads();
        // multipy Asub and Bsub together 
        for (int e = 0;e<BLOCK_SIZE;++e){
            Cvalue += As[row][e] *Bs[e][col];
            __syncthreads();
        }
        // write csub to device memory 
        // each threads writes one element 
        SetElement(Csub,row,col,Cvalue);}
}