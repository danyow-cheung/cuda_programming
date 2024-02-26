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





## 异构编程

<img src = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/heterogeneous-programming.png">

如上图所示，CUDA 编程模型假设 CUDA 线程在物理上独立的*设备*上执行，该设备作为运行 C++ 程序的*主机的协处理器运行。*例如，当内核在 GPU 上执行而 C++ 程序的其余部分在 CPU 上执行时，就会出现这种情况。





CUDA 编程模型还假设主机和设备都在 DRAM 中维护自己独立的内存空间，分别称为*主机内存*和*设备内存*。因此，程序通过调用 CUDA 运行时（在[编程接口](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)中描述）来管理内核可见的全局、常量和纹理内存空间。这包括设备内存分配和释放以及主机和设备内存之间的数据传输。





统一内存提供*托管内存*来桥接主机和设备内存空间。托管内存可作为具有公共地址空间的单个一致内存映像从系统中的所有 CPU 和 GPU 进行访问。此功能可实现设备内存的超额订阅，并且无需在主机和设备上显式镜像数据，从而大大简化移植应用程序的任务。有关统一内存的介绍，请参阅统一内存[编程。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd)

**笔记**

- 串行代码在主机上执行，并行代码在设备上执行



## 异步SIMT编程模型

在 CUDA 编程模型中，线程是执行计算或内存操作的最低抽象级别



异步编程模型定义了与 CUDA 线程相关的异步操作的行为。



异步编程模型定义了用于 CUDA 线程之间同步的[异步屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier)的行为。该模型还解释并定义了如何使用[cuda::memcpy_async在 GPU 中计算时从全局内存异步移动数据。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)



### 异步操作

异步操作被定义为由 CUDA 线程发起并像由另一个线程一样异步执行的操作。在格式良好的程序中，一个或多个 CUDA 线程与异步操作同步。启动异步操作的 CUDA 线程不需要位于同步线程中。



这样的异步线程（as-if 线程）始终与启动异步操作的 CUDA 线程相关联。异步操作使用同步对象来同步操作的完成。这样的同步对象可以由用户显式管理（例如，`cuda::memcpy_async`）或在库内隐式管理（例如，`cooperative_groups::memcpy_async`）。

同步对象可以是 a`cuda::barrier`或 a `cuda::pipeline`。[这些对象在使用 cuda::pipeline 的异步屏障](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier)和[异步数据副本](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-data-copies)中详细解释。这些同步对象可以在不同的线程范围内使用。范围定义了可以使用同步对象来与异步操作同步的线程集。下表定义了 CUDA C++ 中可用的线程范围以及可以与每个线程同步的线程。

| 线程范围                                  | 描述                                                |
| ----------------------------------------- | --------------------------------------------------- |
| `cuda::thread_scope::thread_scope_thread` | 只有启动异步操作的CuDA线程才会同步                  |
| `cuda::thrad_scope::thrad_scope_block`    | 与发起线程同步的同一线程块内的所有或任何cuda线程    |
| `cuda::thread_scope::thread_scope_device` | 与发起线程同步的同一gpu设备中的所有或任何CUDA线程   |
| `cuda::thread_scope::thread_scope_system` | 与发起线程同步的同一系统中的所有或任何cuda或cpu线程 |

上面线程作用域是cuda标准c++



## 计算能力

设备的计算能力由版本号表示，有时也称为“SM 版本” *。*该版本号标识 GPU 硬件支持的功能，并由应用程序在运行时使用来确定当前 GPU 上可用的硬件功能和/或指令。



[支持 CUDA 的 GPU](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-enabled-gpus)列出了所有支持 CUDA 的设备及其计算能力。[计算能力](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)给出了每种计算能力的技术规格。









# 3. 编程接口

CUDA C++ 为熟悉 C++ 编程语言的用户提供了一条简单的途径，可以轻松编写供设备执行的程序。

它由 C++ 语言的最小扩展集和运行时库组成。



核心语言扩展已在[编程模型](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)中引入。它们允许程序员将内核定义为 C++ 函数，并在每次调用该函数时使用一些新语法来指定网格和块维度。[所有扩展的完整描述可以在C++ 语言扩展](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#c-language-extensions)中找到。包含其中一些扩展的任何源文件都必须按照[使用 NVCC 编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compilation-with-nvcc)`nvcc`中所述进行编译。



运行是由`cuda runtime `引入。提供在host上的c和c++函数。以分配和释放设备内存，在主机内存和设备内存之间传输数据，管理具有多个设备的系统



运行时构建在较低级别的 C API（CUDA 驱动程序 API）之上，应用程序也可以访问该 API。驱动程序 API 通过公开较低级别的概念（例如 CUDA 上下文（设备的主机进程的模拟）和 CUDA 模块（设备的动态加载库的模拟））来提供额外的控制级别。大多数应用程序不使用驱动程序 API，因为它们不需要这种额外的控制级别，并且在使用运行时时，上下文和模块管理是隐式的，从而产生更简洁的代码。

## 与NVCC编译

*可以使用称为PTX 的*CUDA 指令集架构编写内核，PTX 参考手册中对此进行了描述。然而，使用高级编程语言（例如 C++）通常更有效。在这两种情况下，**内核都必须编译为二进制代码才能`nvcc`在设备上执行。**



### 编译工作流程

#### 离线编译

使用nvcc編譯的原始檔案可以包括主機程式碼（即，在主機上執行的程式碼）和設備程式碼（即在設備上執行的碼）的混合。 nvcc的基本工作流程包括將設備程式碼與主機程式碼分離，然後：

- 编译设备代码成轉換成彙編形式（PTX程式碼）和/或二進位形式（cubin對象），
- 以及通過替換`<<<…>>`來修改主機程式碼 通過必要的CUDA運行時函數調用在內核中引入的語法（並在執行配寘中詳細描述），以從PTX程式碼和/或cubin對象加載和啟動每個編譯的內核。



修改後的主機程式碼要麼作為C++程式碼輸出，留下來使用另一個工具進行編譯，要麼通過讓nvcc在最後一個編譯階段調用主機編譯器直接作為目標程式碼輸出。

应用接下来能做

- 链接到已编译的主机代码(这是最常见的情况) ,
- 或者忽略修改后的主机代码(如果有的话) ，使用 CUDA 驱动程序 API (参见驱动程序 API)来加载和执行 PTX 代码或 Cubin 对象。

#### 实时编译

应用程序在运行时加载的任何 PTX 代码都由设备驱动程序进一步编译成二进制代码。这叫即时编译。**即时编译增加了应用程序的加载时间，但允许应用程序从每个新设备驱动程序带来的任何新的编译器改进中受益。它也是应用程序在编译时不存在的设备上运行的唯一方法**，详见应用程序兼容性。





当设备驱动程序为某个应用程序即时编译某些 PTX 代码时，它会自<u>动缓存生成的二进制代码的副本</u>，以避免在应用程序的后续调用中重复编译。缓存(称为计算缓存)在设备驱动程序升级时自动失效，因此应用程序可以从设备驱动程序中内置的新的实时编译器的改进中受益。





Environment variables are available to control just-in-time compilation as described in [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)



作为使用 nvcc 编译 CUDA C + + 设备代码的一种替代方法，NVRTC 可以用来在运行时将 CUDA C + + 设备代码编译成 PTX。NVRTC 是 CUDA C + + 的运行时编译库



### 二进制兼容性

二进制代码是特定于体系结构的。Cubin 对象是使用编译器选项-指定目标体系结构的代码生成的: 例如，使用`-code = sm _ 80`进行编译将为具有`8.0计算能力`的设备生成二进制代码。二进制兼容性从一个次要版本到下一个版本得到保证，但是从一个次要版本到上一个版本或者跨主要版本的兼容性得不到保证。换句话说，为计算能力 X.y 生成的 Cubin 对象只能在具有计算能力 X.z 且 z ≥ y 的设备上执行。

> 之前遇到的情况有，驱动版本太高了然后4090算力不够的情况，就需要在~/.bashrc进行配置





**笔记**

**只支持桌面的二进制兼容性。Tegra 不支持它。此外，桌面和 Tegra 之间的二进制兼容性也不受支持。**



### PTX 兼容性

有些 PTX 指令只支持计算能力较高的设备。例如，只有计算能力为5.0及以上的设备才支持 Warp Shuffle 函数。Arch 编译器选项指定在将 C + + 编译为 PTX 代码时假定的计算能力。因此，例如，包含翘曲洗牌的代码必须使用`-arch = computer _ 50`(或更高版本)进行编译。



为某些特定计算能力生成的 PTX 代码总是可以编译成具有更大或相同计算能力的二进制代码。注意，从早期 PTX 版本编译的二进制文件可能不会使用某些硬件特性。例如，由为计算能力6.0(Pascal)生成的 PTX 编译的具有计算能力7.0(Volta)的二进制目标设备将不使用 Tensor Core 指令，因为这些指令在 Pascal 上不可用。因此，最终的二进制文件的性能可能比使用最新版本的 PTX 生成的二进制文件的性能更差。





### 应用兼容性

要在具有特定计算能力的设备上执行代码，应用程序必须加载与此计算能力兼容的二进制代码或 PTX 代码，如二进制兼容性和 PTX 兼容性中所述。特别是，**为了能够在计算能力更强的未来架构上执行代码(目前还不能生成二进制代码) ，应用程序必须加载为这些设备即时编译的 PTX 代码(见即时编译)。**



在 CUDA C + + 应用程序中嵌入哪些 PTX 和二进制代码由-arch 和-code 编译器选项或-gencode 编译器选项控制，详见 nvcc 用户手册。比如说,

```
nvcc x.cu
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_60,code=sm_60
        -gencode arch=compute_70,code=\"compute_70,sm_70\"
        
```



嵌入与计算能力5.0和6.0兼容的二进制代码(第一和第二代码选项)和与计算能力7.0兼容的 PTX 和二进制代码(第三代码选项)。



生成主机代码，以便在运行时自动选择要加载和执行的最合适的代码，在上面的示例中，这些代码将是:

- PTX 代码，在运行时为具有8.0和8.6计算能力的设备编译成二进制代码。
- 计算能力为7.0和7.5的设备的二进制代码,



Cu 可以有一个使用翘曲减少操作的优化代码路径，例如，这些操作只在计算能力为8.0或更高的设备中支持。根据计算能力，可以使用 `_ _ CUDA _ ARCH _ _ `宏区分不同的代码路径。它只为设备代码定义。例如，在使用`-arch = computer _ 80`进行编译时，`_ _ CUDA _ ARCH _ _ `等于800。



使用驱动程序 API 的应用程序必须编译代码来分离文件，并在运行时显式加载和执行最合适的文件。



Volta 架构引入了独立线程调度，它改变了 GPU 上线程调度的方式。对于依赖于以前体系结构中 SIMT 调度的特定行为的代码，独立线程调度可能会改变参与的线程集，从而导致不正确的结果。为了在实现独立线程调度中详细说明的纠正措施的同时帮助迁移，Volta 开发人员可以使用编译器选项组合`-arch = computer _ 60-code = sm _ 70`选择进入 Pascal 的线程调度。



`Nvcc` 用户手册列出了`-arch、-code `和`-gencode `编译器选项的各种简写。例如,`-arch = sm _ 70`是`-arch = computer _ 70-code = computer _ 70`，sm _ 70(与-gencode arch = computer _ 70，code = “ computer _ 70，sm _ 70”相同)的缩写。



### C++ 兼容性

编译器的前端根据 C + + 语法规则处理 CUDA 源文件。主机代码支持完整的 C + + 。但是，如 C + + 语言支持中所述，设备代码只完全支持 C + + 的一个子集。



输入源代码按照 C++ ISO/IEC 14882:2003、C++ ISO/IEC 14882:2011、C++ ISO/IEC 14882:2014 或 C++ ISO/IEC 14882:2017 规范进行处理，CUDA 前端编译器目标模拟与 ISO 规范的任何主机编译器差异。此外，支持的语言使用本文档[13](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#fn13)中描述的 CUDA 特定结构进行扩展，并且受到下述限制。

[C++11 语言功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp11-language-features)、[C++14 语言功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp14-language-features)和[C++17 语言功能](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cpp17-language-features)分别提供 C++11、C++14、C++17 和 C++20 功能的支持矩阵。[限制](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrictions)列出了语言限制。[多态函数包装器](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#polymorphic-function-wrappers)和[扩展 Lambda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda)描述了附加功能。[代码示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#code-samples)提供了代码示例。





### 64位兼容性

64位版本的 nvcc 以64位模式编译设备代码(也就是说，指针是64位的)。以64位模式编译的设备代码只支持以64位模式编译的主机代码。



## CUDA Runtime

运行时在库中实现，<u>该库通过或`cudart`静态链接到应用程序，或者通过或动态链接到应用程序</u>。需要和/或进行动态链接的应用程序通常将它们作为应用程序安装包的一部分。只有在链接到 CUDA 运行时同一实例的组件之间传递 CUDA 运行时符号的地址才是安全的。`cudart.lib``libcudart.a``cudart.dll``libcudart.so``cudart.dll``cudart.so`



它的所有入口点都以 为前缀`cuda`。

正如[异构编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming)中提到的，CUDA 编程模型假设系统由主机和设备组成，每个主机和设备都有自己独立的内存。[设备内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory)概述了用于管理设备内存的运行时函数。

[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)说明了如何使用[线程层次结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)中引入的共享内存来最大限度地提高性能。

[页锁定主机内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory)引入了页锁定主机内存，需要将内核执行与主机和设备内存之间的数据传输重叠。

[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)描述了用于在系统中的各个级别启用异步并发执行的概念和 API。

[多设备系统](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system)展示了编程模型如何扩展到具有连接到同一主机的多个设备的系统。

[错误检查](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking)描述了如何正确检查运行时生成的错误。

[调用堆栈](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#call-stack)提到了用于管理 CUDA C++ 调用堆栈的运行时函数。

[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)呈现纹理和表面内存空间，提供另一种访问设备内存的方式；它们还公开了 GPU 纹理硬件的子集。

[图形互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graphics-interoperability)介绍了运行时提供的各种函数，用于与两个主要图形 API（OpenGL 和 Direct3D）进行互操作。

### 初始化

从 CUDA 12.0 开始，`cudaInitDevice()`和`cudaSetDevice()`调用会初始化与指定设备关联的运行时和主要上下文。如果没有这些调用，<u>运行时将隐式使用设备 0 并根据需要进行自初始化以处理其他运行时 API 请求。在计时运行时函数调用以及解释第一次调用到运行时的错误代码时</u>，需要牢记这一点。在 12.0 之前，`cudaSetDevice()`不会初始化运行时，应用程序通常会使用无操作运行时调用将`cudaFree(0)`运行时初始化与其他 api 活动隔离（都是为了计时和错误处理）



运行时为系统中的每个设备创建一个 CUDA 上下文（有关CUDA 上下文的更多详细信息，请参阅[上下文）。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#context)该上下文是该设备的*主要上下文*，并在第一个运行时函数处初始化，该函数需要该设备上的活动上下文。它在应用程序的所有主机线程之间共享。作为此上下文创建的一部分，如有必要，设备代码将被即时编译（请参阅[即时编译](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#just-in-time-compilation)）并加载到设备内存中。这一切都是透明发生的。例如，如果需要驱动程序 API 互操作性，可以从驱动程序 API 访问设备的主要上下文，如[运行时和驱动程序 API 之间的互操作性](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interoperability-between-runtime-and-driver-apis)中所述。



当主机线程调用 时`cudaDeviceReset()`，这会破坏主机线程当前操作的设备的主要上下文（即，[设备选择](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection)中定义的当前设备）。将此设备作为当前设备的任何主机线程进行的下一个运行时函数调用将为该设备创建一个新的主上下文。



 **笔记**

**CUDA 接口使用全局状态，该状态在主机程序启动期间初始化并在主机程序终止期间销毁。CUDA 运行时和驱动程序无法检测此状态是否无效，因此在程序启动或 main 之后终止期间使用任何这些接口（隐式或显式）将导致未定义的行为。**

**从 CUDA 12.0 开始，`cudaSetDevice()`现在将在更改主机线程的当前设备后显式初始化运行时。以前版本的 CUDA 会延迟新设备上的运行时初始化，直到在`cudaSetDevice()`. 此更改意味着现在检查`cudaSetDevice()`初始化错误的返回值非常重要。**

**参考手册的错误处理和版本管理部分中的运行时函数不会初始化运行时。**



### 设备内存

正如[异构编程](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#heterogeneous-programming)中提到的，CUDA 编程模型假设系统由主机和设备组成，每个主机和设备都有自己独立的内存。内核在设备内存之外运行，因此运行时提供了分配、释放和复制设备内存以及在主机内存和设备内存之间传输数据的函数。

设备内存可以分配为*线性内存*或*CUDA 数组*。



CUDA 数组是针对纹理获取而优化的不透明内存布局。它们在[纹理和表面内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)中进行了描述。

线性内存分配在单个统一地址空间中，这意味着单独分配的实体可以通过指针相互引用，例如在二叉树或链表中。地址空间的大小取决于主机系统（CPU）和所使用的 GPU 的计算能力：





线性内存通常使用 进行分配`cudaMalloc()`和释放`cudaFree()`，并且主机内存和设备内存之间的数据传输通常使用 进行`cudaMemcpy()`。[在Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels)的向量加法代码示例中，需要将向量从主机内存复制到设备内存：

> vecadd_cuda.cpp

`cudaMallocPitch()`线性内存也可以通过和来分配`cudaMalloc3D()`。建议将这些函数用于 2D 或 3D 数组的分配，因为它可以确保分配得到适当的填充以满足[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中描述的对齐要求，从而确保在访问行地址或在 2D 数组与其他区域之间执行复制时获得最佳性能设备内存（使用`cudaMemcpy2D()`和`cudaMemcpy3D()`函数）。返回的间距（或步幅）必须用于访问数组元素。以下代码示例分配一个浮点值的`width`x 2D 数组，并演示如何在设备代码中循环遍历数组元素：`height`

> loop_2dArray.cpp





**笔记**

**为了避免分配过多内存从而影响系统范围的性能，请根据问题大小向用户请求分配参数。如果分配失败，您可以回退到其他较慢的内存类型（`cudaMallocHost()`、`cudaHostRegister()`等），或者返回一个错误，告诉用户需要多少内存但被拒绝。如果您的应用程序由于某种原因无法请求分配参数，我们建议使用`cudaMallocManaged()`支持它的平台。**



参考手册列出了用于在用 分配的线性内存`cudaMalloc()`、用`cudaMallocPitch()`或分配的线性内存`cudaMalloc3D()`、CUDA 数组以及为全局或常量内存空间中声明的变量分配的内存之间复制内存的所有各种函数。

以下代码示例说明了通过运行时 API 访问全局变量的各种方法：

> visit_globalVariable.cpp

`cudaGetSymbolAddress()`用于检索指向为全局内存空间中声明的变量分配的内存的地址。分配的内存大小通过 获得`cudaGetSymbolSize()`。

### 设备内存L2访问管理

当CUDA内核重复访问全局内存中的数据区域时，这种数据访问可以被认为是*持久的*。另一方面，如果数据仅被访问一次，则这种数据访问可以被认为是*流式的*。

从 CUDA 11.0 开始，计算能力 8.0 及以上的设备能够影响 L2 缓存中数据的持久性，从而有可能为全局内存提供更高的带宽和更低的延迟访问。

> 涉及到cuda的显存管理

#### 为持久访问预留L2缓存

L2 高速缓存的一部分可以留出用于对全局内存进行持久数据访问。持久访问优先使用 L2 缓存的这部分预留部分，而对全局内存的正常或流式访问只能在持久访问未使用时才利用 L2 的这部分。

用于持久访问的 L2 缓存预留大小可以在限制范围内进行调整：

> l2_cache.cpp



当有多卡gpu的时候，l2缓存功能会被禁用

使用多进程服务 (MPS) 时，无法通过 更改 L2 缓存预留大小`cudaDeviceSetLimit`。相反，预留大小只能在 MPS 服务器启动时通过环境变量指定`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`。

#### 持久化访问L2策略

**访问策略窗口指定全局内存的连续区域以及 L2 缓存中用于该区域内访问的持久性属性。**

下面的代码示例展示了如何使用 CUDA Stream 设置 L2 持久访问窗口。

> cuda_stream.cpp



该`hitRatio`参数可用于指定接收该`hitProp`属性的访问的比例。在上面的两个示例中，全局内存区域中 60% 的内存访问`[ptr..ptr+num_bytes)`具有持久属性，40% 的内存访问具有流属性。哪些特定内存访问被归类为持久性 (the `hitProp`) 是随机的，概率约为`hitRatio`；概率分布取决于硬件架构和内存范围。





例如，如果 L2 预留缓存大小为 16KB，并且`num_bytes`in`accessPolicyWindow`为 32KB：

- 当 a`hitRatio`为 0.5 时，硬件将随机选择 32KB 窗口中的 16KB 指定为持久并缓存在预留的 L2 缓存区域中。
- 当 a`hitRatio`为 1.0 时，硬件将尝试将整个 32KB 窗口缓存在预留的 L2 缓存区域中。由于预留区域小于窗口，缓存行将被逐出，以将 32KB 数据中最近使用的 16KB 保留在 L2 缓存的预留部分中。

因此，**`hitRatio`可以用来避免缓存行的颠簸，并总体减少移入和移出 L2 缓存的数据量。**

低于 1.0 的值`hitRatio`可用于手动控制与`accessPolicyWindow`并发 CUDA 流不同的数据量可以在 L2 中缓存。例如，设L2预留缓存大小为16KB；两个不同 CUDA 流中的两个并发内核（每个都有 16KB`accessPolicyWindow`且`hitRatio`值均为 1.0）在竞争共享 L2 资源时可能会逐出彼此的缓存行。但是，如果两者`accessPolicyWindows`的 hitRatio 值为 0.5，则它们驱逐自己或彼此的持久缓存行的可能性较小。



#### 访问L2属性

为不同的全局内存数据访问定义了三种类型的访问属性：

1. `cudaAccessPropertyStreaming`：与流属性一起发生的内存访问不太可能保留在 L2 缓存中，因为这些访问会被优先逐出。
2. `cudaAccessPropertyPersisting`：具有持久属性的内存访问更有可能保留在 L2 缓存中，因为这些访问优先保留在 L2 缓存的预留部分中。
3. `cudaAccessPropertyNormal`：此访问属性将先前应用的持久访问属性强制重置为正常状态。来自先前 CUDA 内核的具有持久属性的内存访问可能会在其预期使用后很长时间内保留在 L2 缓存中。这种使用后持久性减少了不使用持久性属性的后续内核可用的 L2 缓存量。使用`cudaAccessPropertyNormal`属性重置访问属性窗口会删除先前访问的持久（优先保留）状态，就好像先前访问没有访问属性一样。
4. 

#### L2 持久化示例

以下示例演示如何为持久访问预留二级缓存，通过 CUDA Stream 在 CUDA 内核中使用预留的二级缓存，然后重置二级缓存。

> l2_Persistenc.cpp



#### 将L2访问重置为正常

来自先前 CUDA 内核的持久 L2 缓存行可能在使用后很长一段时间内仍保留在 L2 中。因此，将 L2 高速缓存重置为正常对于流式或正常存储器访问以利用具有正常优先级的 L2 高速缓存非常重要。可通过三种方式将持久访问重置为正常状态。

1. 使用访问属性重置先前的持久内存区域`cudaAccessPropertyNormal`。
2. 通过调用将所有持久 L2 缓存线重置为正常`cudaCtxResetPersistingL2Cache()`。
3. **最终**未触及的线路会自动重置为正常。强烈建议不要依赖自动重置，因为自动重置发生所需的时间长度不确定。





#### 管理L2预留缓存的使用

在不同 CUDA 流中同时执行的多个 CUDA 内核可能会为其流分配不同的访问策略窗口。然而**，L2 预留缓存部分在所有这些并发 CUDA 内核之间共享。**因此，该预留缓存部分的净利用率是所有并发内核单独使用的总和。当持久访问量超过预留的 L2 高速缓存容量时，将内存访问指定为持久访问的好处就会减弱。

要管理预留 L2 缓存部分的利用率，应用程序必须考虑以下事项：

- L2 预留缓存的大小。
- 可以同时执行的 CUDA 内核。
- 所有可以同时执行的 CUDA 内核的访问策略窗口。
- 何时以及如何需要 L2 重置，以允许正常或流式访问以相同的优先级利用先前预留的 L2 缓存。



#### 查询二级缓存属性

与L2缓存相关的属性是struct的一部分`cudaDeviceProp`，可以使用CUDA运行时API进行查询`cudaGetDeviceProperties`

CUDA 设备属性包括：

- `l2CacheSize`：GPU 上可用的二级缓存数量。
- `persistingL2CacheMaxSize`：可以为持久内存访问预留的 L2 缓存的最大数量。
- `accessPolicyMaxWindowSize`：访问策略窗口的最大大小。





#### 控制持久内存访问的L2缓存预留大小

使用 CUDA 运行时 API 查询用于持久内存访问的 L2 预留缓存大小`cudaDeviceGetLimit`，并使用 CUDA 运行时 API 将其设置`cudaDeviceSetLimit`为`cudaLimit`. 设置此限制的最大值为`cudaDeviceProp::persistingL2CacheMaxSize`。

```c++
enum cudaLimit{
	/* other fields not shown*/
  cudaLimitPersistingL2CacheSize
}

```



### 共享内存

如[变量内存空间说明符](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#variable-memory-space-specifiers)中详细描述的，共享内存是使用`__shared__`内存空间说明符分配的。

[正如线程层次结构](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)中提到的和[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)中详细介绍的那样，共享内存预计将比全局内存快得多。它可以用作暂存存储器（或软件管理的缓存），以最大限度地减少来自 CUDA 块的全局存储器访问，如以下矩阵乘法示例所示。

以下代码示例是矩阵乘法的简单实现，不利用共享内存。每个线程读取*A*的一行和*B的一列，并计算*C*的相应元素，如图[8](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-matrix-multiplication-no-shared-memory)所示。*因此， A*从全局内存中读取*B.width*次，*B*被读取*A.height*次。

> shared_cache.cpp



<img src = 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png'>

以下代码示例是利用共享内存的矩阵乘法的实现。在此实现中，每个线程块负责计算*C*的一个方子矩阵*Csub*，并且块内的每个线程负责计算*Csub*的一个元素。如下图所示，*Csub*等于两个矩形矩阵的乘积：维度为( *A.width, block_size ) 的**A*子矩阵，其行索引与*Csub*相同，维度为*B*的子矩阵( *block_size, A.width ) 与**Csub*具有相同的列索引。为了适应设备的资源，这两个矩形矩阵根据需要被划分为尽可能多的维度为*block_size的方阵，并且*Csub*被计算为这些方阵的乘积之和。这些乘积中的每一个都是通过以下方式执行的：首先将两个相应的方阵从全局内存加载到共享内存，并用一个线程加载每个矩阵的一个元素，然后让每个线程计算乘积的一个元素。每个线程将每个乘积的结果累积到寄存器中，完成后将结果写入全局内存。

> shared_cache_muplity.cpp

通过以这种方式分块计算，我们可以利用快速共享内存并节省大量全局内存带宽，因为*A*仅从全局内存中读取 ( *B.width / block_size ) 次，而**B*则被读取 ( *A.height / block_size* ) 次。

先前代码示例中的 Matrix 类型通过步幅字段进行了增强，*以便*可以使用相同类型有效地表示子矩阵*。*[__device__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-function-specifier)函数用于获取和设置元素以及从矩阵构建任何子矩阵

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png'>