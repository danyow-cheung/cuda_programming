GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;
int main(){
    //initialize opengl and glut for device 0 
    //and make the opengl context current 
    ...
    glutDisplayFunc(display);

    //explicitly set device 0 
    cudaSetDevice(0);

    //create buffer object and register it with cuda 
    glGenBuffers(1,&positionsVBO);
    glBinfBuffer(GL_ARRAY_BUFFER,positionsVBO);
    unsigned int size = width * height *4 *sizeof(float);
    glBufferData(GL_ARRAY_BUFFER,size,0,GL_DYNAMIC_DRAW);
    glBindBUffer(GL_ARRAY_BUFFER,0);

    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,
                        positionsVBO,cudaGraphicsMapFlagsWriteDiscard);

    //launch rendering loop
    glutMainLoop();
    ...
}
void display(){
    // map buffer object for writing from cuda
    flaot4* positions;
    cudaGraphicsMapResources(1,&positionsVBO_CUDA,0);

    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,&num_bytes,positionsVBO_CUDA);

    //execute kernel 
    dim3 dimBlock(16,16,1);
    dim3 dimGrid(width/dimBlock.x,height/dimBlock.y,1);
    createVerticess<<<dimGrid,dimBlock>>>(postions,time,width,height);

    //unmap buffer object 
    cudaGraphicsUnmapResources(1,&positionsVBO_CUDA,0);
    //render from buffer object 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);

    glVertexPointer(4,GL_FLOAT,0,0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS,0,width*height);
    glDisableClientState(GL_VERTEX_ARRAY);

    //swap buffers 
    glutSwapBuffers();
    glutPostRedisplay();
}

void deleteVBO(){
    cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    glDeleteBuffers(1,&positionsVBO);
}

__global__ void createVerticess(float4* positions,float time,unsigned int width,
                                unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y ;

    //calculatte uv coordinates 
    float u = x / (float) width;
    float v = y / (float) height;

    u = u*0.2f -1.0f ;
    v = v*0.2f - 1.0f;

    //calculate simple sine wave patteren 
    float freq = 4.0f ;
    float w = sinf(u*freq+time)*cosf(v*freq+time) *0.5f;

    //write postions 
    positions[y*width+x] = make_float4(u,w,v,1.0f);
}

