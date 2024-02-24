# 2.Programming model
## Kernels
CUDA C++ 通过允许程序员定义称为内核的 C++ 函数来扩展 C++，这些函数在调用时由 N 个不同的CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

内核是使用声明说明符定义的，并且使用新的执行配置`__global__`语法指定为给定内核调用执行该内核的 CUDA 线程数（请参阅C++ 语言扩展）。每个执行内核的线程都会被赋予一个唯一的线程 ID，该 ID 可以在内核中通过内置变量进行访问。`<<<...>>>`

作为说明，以下示例代码使用内置变量threadIdx，将两个大小为N的向量A和B相加，并将结果存储到向量C中：
```c++
// kerenl defintion 
__global__ void VecAdd(float*a,float*b,float*c){
  int i = threadIdx.x;
  c[i] = a[i]+b[i];
}
int main(){
  // kernel invocation with N threads
  VecAdd<<<1,N>>>(a,b,c);
}
```

执行N个`Vec Add()`线程中的每一个都执行一次向量加法



## Thread Hierarchy

为了方便起见，`threadIdx`是一个3分量向量，因此可以使用一维、二维或三维线程*索引*来标识线程，形成一维、二维或三维线程块，称为*线程块*。这提供了一种自然的方式来调用域中元素（例如向量、矩阵或体积）的计算。

线程的索引和线程 ID 之间的关系非常简单：对于一维块，它们是相同的；对于一维块，它们是相同的；对于一维块，它们是相同的。*对于大小为(Dx, Dy)*的二维块，索引为*(x, y)*的线程的线程 ID为*(x + y Dx)*；*对于大小为(Dx, Dy, Dz)*的三维块，索引为*(x, y, z)*的线程的线程 ID为*(x + y Dx + z Dx Dy)*。

例如，以下代码将大小为*NxN的两个矩阵**A*和*B*相加，并将结果存储到矩阵*C*中：

```c++
// kerenl defintion 
__global__ void MatAdd(float A[N][N],float B[N][N],float c[N][N]){
	int i = threadIdx.x;
  int j = threadIdx.y;
  c[i][j] = A[i][j] + B[i][j];
}

int main(){
  // kernel invocation with one block of N*N*1 threads 
  int numBlocks = 1;
  dim3 threadsPerBlock(N,N);
  MatAdd<<numBlocks,threadPerBlock>>(A,B,C);
  ...
}
```

每个块的线程数量是有限的，因为块中的所有线程都应驻留在同一个流式多处理器核心上，并且必须共享该核心的有限内存资源。在当前的 GPU 上，一个线程块最多可以包含 1024 个线程。



然而，一个内核可以由多个形状相同的线程块来执行，因此线程总数等于每个块的线程数乘以块数。

块被组织成一维、二维或三维线程块*网格，如图*[4](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy-grid-of-thread-blocks)所示。网格中线程块的数量通常由正在处理的数据大小决定，该数据通常超过系统中处理器的数量。

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-thread-blocks.png'>



语法中指定的每个块的线程数和每个网格的块数可以是或`<<<...>>>`类型。可以如上例所示指定二维块或网格。`int``dim3`

**网格内的每个块都可以由一维、二维或三维唯一索引来标识，该索引可通过内置`blockIdx`变量在内核中访问。线程块的维度可以在内核中通过内置`blockDim`变量访问。**

扩展前面的`MatAdd()`示例以处理多个块，代码如下。

> vecadd.cpp







16x16（256 个线程）的线程块大小虽然在本例中是任意的，但却是常见的选择。网格是用足够的块创建的，以便像以前一样每个矩阵元素有一个线程。为简单起见，此示例假设每个维度中每个网格的线程数可被该维度中每个块的线程数整除，但情况不一定如此。

线程块需要独立执行：必须能够以任何顺序（并行或串行）执行它们。[这种独立性要求允许在任意数量的内核上以任意顺序调度线程块，如图 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#scalable-programming-model-automatic-scalability)所示，从而使程序员能够编写随内核数量扩展的代码。

块内的线程可以通过某些*共享内存*共享数据并同步其执行来协调内存访问来进行协作。更准确地说，可以通过调用`__syncthreads()`内部函数来指定内核中的同步点；`__syncthreads()`充当屏障，块中的所有线程必须等待才能继续进行。[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)给出了使用共享内存的示例。除了 之外`__syncthreads()`，[协作组 API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)还提供了一组丰富的线程同步原语。

为了高效协作，共享内存预计是每个处理器核心附近的低延迟内存（很像 L1 缓存），并且`__syncthreads()`预计是轻量级的。





### Thread Block Clusters

[随着 NVIDIA计算能力 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-9-0)的推出，CUDA 编程模型引入了一个可选的层次结构级别，称为由线程块组成的线程块集群。与如何保证线程块中的线程在流式多处理器上共同调度类似，集群中的线程块也保证在 GPU 中的 GPU 处理集群 (GPC) 上共同调度。

与线程块类似，簇也被组织成一维、二维或三维，如图[5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters-grid-of-clusters)所示。簇中线程块的数量可以由用户定义，并且簇中最多支持 8 个线程块作为 CUDA 中的可移植簇大小。请注意，在 GPU 硬件或 MIG 配置上太小而无法支持 8 个多处理器时，最大集群大小将相应减小。这些较小配置以及支持超过 8 的线程块簇大小的较大配置的标识是特定于体系结构的，并且可以使用 API 进行查询`cudaOccupancyMaxPotentialClusterSize`。

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png'>

可以使用编译器时内核属性`__cluster_dims__(X,Y,Z)`或使用 CUDA 内核启动 API在内核中启用线程块簇`cudaLaunchKernelEx`。下面的示例展示了如何使用编译器时内核属性启动集群。使用内核属性的簇大小在编译时固定，然后可以使用经典的. 如果内核使用编译时簇大小，则在启动内核时无法修改簇大小。`<<< , >>>`

> cluster.cpp





线程块簇大小也可以在运行时设置，并且可以使用 CUDA 内核启动 API 来启动内核`cudaLaunchKernelEx`。下面的代码示例展示了如何使用可扩展 API 启动集群内核。

> lanuch.cpp



在计算能力9.0的GPU中，集群中的所有线程块都保证在单个GPU处理集群（GPC）上共同调度，并允许集群中的线程块使用Cluster Group API执行硬件支持[的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg)同步`cluster.sync()`。`num_threads()`集群组还提供成员函数，分别使用和API以线程数或块数查询集群组大小`num_blocks()`。`dim_threads()`可以分别使用和API查询集群组中线程或块的排名`dim_blocks()`。



属于集群的线程块可以访问分布式共享内存。集群中的线程块能够对分布式共享内存中的任何地址进行读取、写入和执行原子操作。[分布式共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#distributed-shared-memory)给出了在分布式共享内存中执行直方图的示例。





## Memory Hierarchy

内存层次结构



CUDA 线程在执行期间可以访问多个内存空间中的数据，如图[6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy-memory-hierarchy-figure)所示<u>。每个线程都有私有本地内存。</u>每个线程块都有对该块的所有线程可见的共享内存，并且与该块具有相同的生命周期。线程块簇中的线程块可以对彼此的共享内存执行读、写和原子操作。所有线程都可以访问相同的全局内存。

还有两个可供所有线程访问的附加只读内存空间：<u>常量内存空间和纹理内存空间</u>。全局、常量和纹理内存空间针对不同的内存使用情况进行了优化（请参阅[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)）。纹理内存还为某些特定的数据格式提供不同的寻址模式以及数据过滤（请参阅[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)）。

<u>全局、常量和纹理内存空间在同一应用程序的内核启动过程中是持久的。</u>

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/memory-hierarchy.png'>