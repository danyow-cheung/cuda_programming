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
    
}