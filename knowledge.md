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

> vecadd.cu







16x16（256 个线程）的线程块大小虽然在本例中是任意的，但却是常见的选择。网格是用足够的块创建的，以便像以前一样每个矩阵元素有一个线程。为简单起见，此示例假设每个维度中每个网格的线程数可被该维度中每个块的线程数整除，但情况不一定如此。

线程块需要独立执行：必须能够以任何顺序（并行或串行）执行它们。[这种独立性要求允许在任意数量的内核上以任意顺序调度线程块，如图 3](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#scalable-programming-model-automatic-scalability)所示，从而使程序员能够编写随内核数量扩展的代码。

块内的线程可以通过某些*共享内存*共享数据并同步其执行来协调内存访问来进行协作。更准确地说，可以通过调用`__syncthreads()`内部函数来指定内核中的同步点；`__syncthreads()`充当屏障，块中的所有线程必须等待才能继续进行。[共享内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)给出了使用共享内存的示例。除了 之外`__syncthreads()`，[协作组 API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)还提供了一组丰富的线程同步原语。

为了高效协作，共享内存预计是每个处理器核心附近的低延迟内存（很像 L1 缓存），并且`__syncthreads()`预计是轻量级的。





### Thread Block Clusters

[随着 NVIDIA计算能力 9.0](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-9-0)的推出，CUDA 编程模型引入了一个可选的层次结构级别，称为由线程块组成的线程块集群。与如何保证线程块中的线程在流式多处理器上共同调度类似，集群中的线程块也保证在 GPU 中的 GPU 处理集群 (GPC) 上共同调度。

与线程块类似，簇也被组织成一维、二维或三维，如图[5](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters-grid-of-clusters)所示。簇中线程块的数量可以由用户定义，并且簇中最多支持 8 个线程块作为 CUDA 中的可移植簇大小。请注意，在 GPU 硬件或 MIG 配置上太小而无法支持 8 个多处理器时，最大集群大小将相应减小。这些较小配置以及支持超过 8 的线程块簇大小的较大配置的标识是特定于体系结构的，并且可以使用 API 进行查询`cudaOccupancyMaxPotentialClusterSize`。

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/grid-of-clusters.png'>

可以使用编译器时内核属性`__cluster_dims__(X,Y,Z)`或使用 CUDA 内核启动 API在内核中启用线程块簇`cudaLaunchKernelEx`。下面的示例展示了如何使用编译器时内核属性启动集群。使用内核属性的簇大小在编译时固定，然后可以使用经典的. 如果内核使用编译时簇大小，则在启动内核时无法修改簇大小。`<<< , >>>`

> cluster.cu





线程块簇大小也可以在运行时设置，并且可以使用 CUDA 内核启动 API 来启动内核`cudaLaunchKernelEx`。下面的代码示例展示了如何使用可扩展 API 启动集群内核。

> lanuch.cu



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

> vecadd_cuda.cu

`cudaMallocPitch()`线性内存也可以通过和来分配`cudaMalloc3D()`。建议将这些函数用于 2D 或 3D 数组的分配，因为它可以确保分配得到适当的填充以满足[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中描述的对齐要求，从而确保在访问行地址或在 2D 数组与其他区域之间执行复制时获得最佳性能设备内存（使用`cudaMemcpy2D()`和`cudaMemcpy3D()`函数）。返回的间距（或步幅）必须用于访问数组元素。以下代码示例分配一个浮点值的`width`x 2D 数组，并演示如何在设备代码中循环遍历数组元素：`height`

> loop_2dArray.cu





**笔记**

**为了避免分配过多内存从而影响系统范围的性能，请根据问题大小向用户请求分配参数。如果分配失败，您可以回退到其他较慢的内存类型（`cudaMallocHost()`、`cudaHostRegister()`等），或者返回一个错误，告诉用户需要多少内存但被拒绝。如果您的应用程序由于某种原因无法请求分配参数，我们建议使用`cudaMallocManaged()`支持它的平台。**



参考手册列出了用于在用 分配的线性内存`cudaMalloc()`、用`cudaMallocPitch()`或分配的线性内存`cudaMalloc3D()`、CUDA 数组以及为全局或常量内存空间中声明的变量分配的内存之间复制内存的所有各种函数。

以下代码示例说明了通过运行时 API 访问全局变量的各种方法：

> visit_globalVariable.cu

`cudaGetSymbolAddress()`用于检索指向为全局内存空间中声明的变量分配的内存的地址。分配的内存大小通过 获得`cudaGetSymbolSize()`。

### 设备内存L2访问管理

当CUDA内核重复访问全局内存中的数据区域时，这种数据访问可以被认为是*持久的*。另一方面，如果数据仅被访问一次，则这种数据访问可以被认为是*流式的*。

从 CUDA 11.0 开始，计算能力 8.0 及以上的设备能够影响 L2 缓存中数据的持久性，从而有可能为全局内存提供更高的带宽和更低的延迟访问。

> 涉及到cuda的显存管理

#### 为持久访问预留L2缓存

L2 高速缓存的一部分可以留出用于对全局内存进行持久数据访问。持久访问优先使用 L2 缓存的这部分预留部分，而对全局内存的正常或流式访问只能在持久访问未使用时才利用 L2 的这部分。

用于持久访问的 L2 缓存预留大小可以在限制范围内进行调整：

> l2_cache.cu



当有多卡gpu的时候，l2缓存功能会被禁用

使用多进程服务 (MPS) 时，无法通过 更改 L2 缓存预留大小`cudaDeviceSetLimit`。相反，预留大小只能在 MPS 服务器启动时通过环境变量指定`CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`。

#### 持久化访问L2策略

**访问策略窗口指定全局内存的连续区域以及 L2 缓存中用于该区域内访问的持久性属性。**

下面的代码示例展示了如何使用 CUDA Stream 设置 L2 持久访问窗口。

> cuda_stream.cu



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

> l2_Persistenc.cu



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

> shared_cache.cu



<img src = 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-without-shared-memory.png'>

以下代码示例是利用共享内存的矩阵乘法的实现。在此实现中，每个线程块负责计算*C*的一个方子矩阵*Csub*，并且块内的每个线程负责计算*Csub*的一个元素。如下图所示，*Csub*等于两个矩形矩阵的乘积：维度为( *A.width, block_size ) 的**A*子矩阵，其行索引与*Csub*相同，维度为*B*的子矩阵( *block_size, A.width ) 与**Csub*具有相同的列索引。为了适应设备的资源，这两个矩形矩阵根据需要被划分为尽可能多的维度为*block_size的方阵，并且*Csub*被计算为这些方阵的乘积之和。这些乘积中的每一个都是通过以下方式执行的：首先将两个相应的方阵从全局内存加载到共享内存，并用一个线程加载每个矩阵的一个元素，然后让每个线程计算乘积的一个元素。每个线程将每个乘积的结果累积到寄存器中，完成后将结果写入全局内存。

> shared_cache_muplity.cu

通过以这种方式分块计算，我们可以利用快速共享内存并节省大量全局内存带宽，因为*A*仅从全局内存中读取 ( *B.width / block_size ) 次，而**B*则被读取 ( *A.height / block_size* ) 次。

先前代码示例中的 Matrix 类型通过步幅字段进行了增强，*以便*可以使用相同类型有效地表示子矩阵*。*[__device__](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-function-specifier)函数用于获取和设置元素以及从矩阵构建任何子矩阵

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/matrix-multiplication-with-shared-memory.png'>





### 分布式共享内存

计算能力9.0中引入的线程块集群为线程块集群中的线程提供了访问集群中所有参与线程块的共享内存的能力。<u>这种分区的共享内存称为*分布式共享内存*，对应的地址空间称为分布式共享内存地址空间。</u>属于线程块簇的线程可以在分布式地址空间中读取、写入或执行原子操作，无论该地址属于本地线程块还是远程线程块。无论内核是否使用分布式共享内存，共享内存大小规范（静态或动态）仍然是每个线程块。分布式共享内存的大小就是每个集群的线程块数量乘以每个线程块的共享内存大小。

访问分布式共享内存中的数据需要所有线程块都存在。用户可以保证所有线程块都已开始使用`cluster.sync()`Cluster [Group](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group-cg) API 执行。用户还需要确保所有分布式共享内存操作都发生在线程块退出之前，例如，如果远程线程块正在尝试读取给定线程块的共享内存，则用户需要确保远程线程块读取的共享内存线程块完成后才能退出。

CUDA 提供了一种访问分布式共享内存的机制，应用程序可以从利用其功能中受益。让我们看一下简单的直方图计算以及如何使用线程块集群在 GPU 上对其进行优化。计算直方图的标准方法是在每个线程块的共享内存中进行计算，然后执行全局内存原子。这种方法的一个限制是共享内存容量。一旦直方图箱不再适合共享内存，用户就需要直接计算直方图，从而计算全局内存中的原子。通过分布式共享内存，CUDA 提供了一个中间步骤，根据直方图箱的大小，可以直接在共享内存、分布式共享内存或全局内存中计算直方图。

下面的 CUDA 内核示例展示了如何根据直方图箱的数量计算共享内存或分布式共享内存中的直方图。

> shared_memory_distributed.cu

上述内核可以在运行时启动，其集群大小取决于所需的分布式共享内存的数量。如果直方图小到足以容纳一个块的共享内存，则用户可以启动集群大小为 1 的内核。下面的代码片段显示了如何根据共享内存需求动态启动集群内核。

> shared_memory_distributed_dyna.cu





### 页锁定主机内存

运行时提供了允许使用*页面锁定*（也称为*固定*）主机内存（而不是由 分配的常规可分页主机内存`malloc()`）的函数：

- `cudaHostAlloc()`分配`cudaFreeHost()`和释放页面锁定主机内存；
- `cudaHostRegister()`页面锁定分配的内存范围`malloc()`（有关限制，请参阅参考手册）。

使用页锁定主机内存有几个好处：

- 对于某些设备，页锁定主机内存和设备内存之间的复制可以与内核执行同时执行，如[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)中所述。
- 在某些设备上，页锁定主机内存可以映射到设备的地址空间，从而无需将其复制到设备内存或从设备内存复制，如[映射内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory)中详细介绍的。
- 在具有前端总线的系统上，如果主机内存被分配为页锁定，则主机内存和设备内存之间的带宽会更高，如果另外将其分配为写组合（如写入组合内存中所述），则主机内存和设备[内存](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#write-combining-memory)之间的带宽会更高。



**笔记**

**页面锁定主机内存不会缓存在非 I/O 一致 Tegra 设备上。此外，`cudaHostRegister()`非 I/O 相干 Tegra 设备也不支持。**





#### 写组合存储器

默认情况下，页锁定主机内存被分配为可缓存的。可以选择将其分配为*写*组合，而不是通过将标志传递`cudaHostAllocWriteCombined`给`cudaHostAlloc()`. 写入组合内存可释放主机的 L1 和 L2 缓存资源，从而为应用程序的其余部分提供更多缓存。此外，在通过 PCI Express 总线传输期间，写组合内存不会被监听，这可以将传输性能提高高达 40%。

从主机读取写组合内存的速度非常慢，因此写组合内存通常应用于主机仅写入的内存。

应避免在 WC 内存上使用 CPU 原子指令，因为并非所有 CPU 实现都能保证该功能。



#### 映射内存

`cudaHostAllocMapped`通过将标志传递给`cudaHostAlloc()`或 将标志传递`cudaHostRegisterMapped`给，还可以将页锁定主机内存块映射到设备的地址空间`cudaHostRegister()`。因此，这样的块通常具有两个地址：一个位于主机存储器中，由`cudaHostAlloc()`或返回`malloc()`，另一个位于设备存储器中，可以使用该地址进行检索`cudaHostGetDevicePointer()`，然后用于从内核内部访问该块。唯一的例外是使用`cudaHostAlloc()`统一地址空间分配的指针和将统一地址空间用于主机和设备（如[统一虚拟地址空间](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-virtual-address-space)中所述）的情况。

直接从内核内部访问主机内存并不能提供与设备内存相同的带宽，但确实有一些优点：

- 无需在设备内存中分配块并在该块与主机内存中的块之间复制数据；数据传输根据内核的需要隐式执行；
- 无需使用流（请参阅[并发数据传输](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-data-transfers)）来将数据传输与内核执行重叠；内核发起的数据传输自动与内核执行重叠。

然而，由于映射的页锁定内存在主机和设备之间共享，因此应用程序必须使用流或事件同步内存访问（请参阅[异步并发执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)），以避免任何潜在的先读后写、先写后读或先写后写的情况。 -写出危险。

为了能够检索指向任何映射的页锁定内存的设备指针，必须在执行任何其他 CUDA 调用之前通过使用`cudaSetDeviceFlags()`该标志进行调用来启用页锁定内存映射。`cudaDeviceMapHost`否则，`cudaHostGetDevicePointer()`将返回错误。

`cudaHostGetDevicePointer()`如果设备不支持映射的页锁定主机内存，也会返回错误。`canMapHostMemory`应用程序可以通过检查设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration)）来查询此功能，对于支持映射页锁定主机内存的设备，该属性等于 1。

请注意，从主机或其他设备的角度来看，在映射的页锁定内存上操作的原子函数（请参阅[原子函数）不是原子的。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)

另请注意，CUDA 运行时要求从主机和其他设备的角度来看，从设备发起的对主机内存的 1 字节、2 字节、4 字节和 8 字节自然对齐加载和存储被保留为单次访问。设备。在某些平台上，内存原子可能会被硬件分解为单独的加载和存储操作。这些组件加载和存储操作对于保留自然对齐的访问具有相同的要求。例如，CUDA 运行时不支持 PCI Express 总线拓扑，其中 PCI Express 桥将设备和主机之间自然对齐的 8 字节写入拆分为两个 4 字节写入。



### 内存同步域

#### 内存栅栏干扰

由于内存栅栏/刷新操作等待的事务数量多于 CUDA 内存一致性模型所需的事务数量，某些 CUDA 应用程序可能会出现性能下降。



这有时会导致干扰：因为 GPU 正在等待源级别不需要的内存操作，所以栅栏/刷新可能需要比必要的时间更长的时间。

请注意，栅栏可能会作为代码中的内在函数或原子显式出现（如示例中所示），也可能隐式出现以在任务边界处实现*同步*关系。

一个常见的例子是，内核在本地 GPU 内存中执行计算，并且并行内核（例如来自 NCCL）正在与对等方执行通信。完成后，本地内核将隐式刷新其写入以满足*与下游工作的任何同步*关系。这可能会不必要地完全或部分地等待来自通信内核的较慢的 nvlink 或 PCIe 写入。



#### 使用域隔离流量

从 Hopper 架构 GPU 和 CUDA 12.0 开始，内存同步域功能提供了一种减轻此类干扰的方法。为了换取代码的显式帮助，GPU 可以通过栅栏操作减少网络投射。每个内核启动都会被赋予一个域 ID。写入和栅栏都用 ID 标记，并且栅栏只会订购与栅栏域匹配的写入。在并发计算与通信示例中，通信内核可以放置在不同的域中。

使用域时，代码必须遵守以下规则：**同一 GPU 上不同域之间的排序或同步需要系统范围的防护**。在域内，设备范围的防护仍然足够。这对于累积性是必要的，因为一个内核的写入不会被另一域中的内核发出的栅栏所包围。本质上，通过确保跨域流量提前刷新到系统范围来满足累积性。

请注意，这会修改 的定义`thread_scope_device`。但是，由于内核将默认为域 0（如下所述），因此可以保持向后兼容性。



#### 在cuda中使用域

可通过新的启动属性`cudaLaunchAttributeMemSyncDomain`和`cudaLaunchAttributeMemSyncDomainMap`. 前者在逻辑域`cudaLaunchMemSyncDomainDefault`和之间进行选择`cudaLaunchMemSyncDomainRemote`，后者提供从逻辑域到物理域的映射。远程域适用于执行远程内存访问的内核，以便将其内存流量与本地内核隔离。但请注意，特定域的选择不会影响内核可以合法执行的内存访问。

域计数可以通过设备属性查询`cudaDevAttrMemSyncDomainCount`。Hopper 有 4 个域。为了促进可移植代码，域功能可以在所有设备上使用，并且 CUDA 将在 Hopper 之前报告计数 1。

拥有逻辑域可以简化应用程序的组合。在堆栈中的低级别（例如从 NCCL）启动的单个内核可以选择语义逻辑域，而无需关心周围的应用程序体系结构。更高级别可以使用映射来引导逻辑域。如果未设置，则逻辑域的默认值为默认域，默认映射是将默认域映射到 0，将远程域映射到 1（在具有超过 1 个域的 GPU 上）。在 CUDA 12.0 及更高版本中，特定库可能会使用远程域来标记启动；例如，NCCL 2.16 就会这样做。总之，这为常见应用程序提供了一种开箱即用的有益使用模式，无需在其他组件、框架或应用程序级别进行代码更改。另一种使用模式（例如在使用 nvshmem 或没有明确区分内核类型的应用程序中）可能是对并行流进行分区。流A可以将两个逻辑域映射到物理域0，流B映射到1，等等。

```c++
// example of launching a kernel with the remote logical domain 
cudaLaunchAttribute domainAttr;
domainAttr.id = cudaLaunchAttrMemSyncDomain;
domainAttr.val = cudaLaunchMemSyncDomainRemote;
cudaLaunchConfig_t config;
// fill out other config fields
config.attrs = &domainAttr;
config.numAttrs = 1;
cudaLaunchKernelEx(&config,mykernel,kernelArg1,kernelArg2...);

// example of settting a mapping for a stream
// (This mapping is the default for streams starting on Hopper if not explicitly set ,and provided for illustration)
cudaLaunchAttributeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(stream,cudaLaunchAttrMemSyncDomainMap,&mapAttr);


// example of mapping different streams to different physical domains ,ignoring logical domain settings 
cudaLaunchAttributeeValue mapAttr;
mapAttr.memSyncDomainMap.default_ = 0;
mapAttr.memSyncDomainMap.remote = 0;
cudaStreamSetAttribute(streamA, cudaLaunchAttrMemSyncDomainMap, &mapAttr);
mapAttr.memSyncDomainMap.default_ = 1;
mapAttr.memSyncDomainMap.remote = 1;
cudaStreamSetAttribute(streamB, cudaLaunchAttrMemSyncDomainMap, &mapAttr);
```

`cudaLaunchKernelEx`与其他启动属性一样，这些属性在 CUDA 流、使用 的单独启动以及 CUDA 图中的内核节点上统一公开。典型的使用将在流级别设置映射并在启动级别设置逻辑域（或将流使用的一部分括起来），如上所述。

在流捕获期间，这两个属性都会复制到图形节点。图从节点本身获取这两个属性，本质上是指定物理域的间接方式。在图启动的流上设置的域相关属性不会在图的执行中使用。





### 异步并发执行

CUDA 将以下操作公开为可以相互并发操作的独立任务：

- 在主机上计算；
- 设备上的计算；
- 内存从主机传输到设备；
- 内存从设备传输到主机；
- 给定设备内存内的内存传输；
- 设备之间的内存传输。

这些操作之间实现的并发级别将取决于设备的功能集和计算能力，如下所述。



#### 主机和设备之间的并发执行

通过异步库函数促进并发主机执行，这些函数在设备完成请求的任务之前将控制权返回给主机线程。使用异步调用，许多设备操作可以一起排队，以便在适当的设备资源可用时由 CUDA 驱动程序执行。这减轻了主机线程管理设备的大部分责任，使其可以自由地执行其他任务。以下设备操作相对于主机是异步的：

- 内核启动；
- 单个设备内存中的内存复制；
- 将 64 KB 或更小的内存块从主机复制到设备；
- 由后缀为`Async`;的函数执行的内存复制
- 内存设置函数调用。

程序员可以通过将环境变量设置为 1 来全局禁用系统上运行的所有 CUDA 应用程序的内核启动异步性。`CUDA_LAUNCH_BLOCKING`此功能仅用于调试目的，不应用作使生产软件可靠运行的方法。

如果通过分析器（Nsight、Visual Profiler）收集硬件计数器，则内核启动是同步的，除非启用了并发内核分析。`Async`如果内存副本涉及未页面锁定的主机内存，那么它们也可能是同步的。

#### 并发内核执行

某些计算能力为 2.x 及更高版本的设备可以同时执行多个内核。`concurrentKernels`应用程序可以通过检查设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration)）来查询此功能，对于支持它的设备，该属性等于 1。

设备可以同时执行的内核启动的最大数量取决于其计算能力，如[表 18](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications-technical-specifications-per-compute-capability)所示。

来自一个 CUDA 上下文的内核无法与来自另一 CUDA 上下文的内核同时执行。GPU 可以对每个上下文进行时间切片以提供前进进度。如果用户想要在 SM 上同时运行多个进程的内核，则必须启用 MPS。

使用许多纹理或大量本地内存的内核不太可能与其他内核同时执行。



#### 数据传输和内核执行的重叠

某些设备可以在内核执行的同时执行与 GPU 之间的异步内存复制。`asyncEngineCount`应用程序可以通过检查设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration)）来查询此功能，对于支持它的设备，该属性大于零。如果复制涉及主机内存，则必须对其进行页面锁定。

还可以与内核执行（在支持`concurrentKernels`设备属性的设备上）和/或与设备之间的复制（对于支持该`asyncEngineCount`属性的设备）同时执行设备内复制。设备内复制是使用标准内存复制功能启动的，目标地址和源地址驻留在同一设备上。

#### 并发数据传输

某些计算能力为 2.x 及更高版本的设备可以重叠传入和传出设备的副本。`asyncEngineCount`应用程序可以通过检查设备属性（请参阅[设备枚举](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-enumeration)）来查询此功能，对于支持它的设备，该属性等于 2。为了实现重叠，传输中涉及的任何主机内存都必须被页面锁定。

#### 流

*应用程序通过流*管理上述并发操作。流是按顺序执行的命令序列（可能由不同的主机线程发出）。另一方面，不同的流可能会相互乱序或同时执行命令；无法保证此行为，因此不应依赖其正确性（例如，内核间通信未定义）。当满足命令的所有依赖性时，可以执行在流上发出的命令。依赖项可以是先前在同一流上启动的命令或来自其他流的依赖项。同步调用的成功完成保证了所有启动的命令都已完成。



##### 创建和销毁

> cuda_stream_create_delete.cu



##### 默认码流

`<->`未指定任何流参数或等效地将流参数设置为零的内核启动和主机设备内存副本将发布到默认流。因此它们是按顺序执行的。



对于使用编译标志编译的代码（或者在包含 CUDA 标头 (和) 之前定义宏的代码），默认流是常规流，并且每个主机线程都有自己的默认流。`--default-stream per-thread``CUDA_API_PER_THREAD_DEFAULT_STREAM``cuda.h``cuda_runtime.h`

**笔记**

**#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1``nvcc`当代码由隐`nvcc`式包含`cuda_runtime.h`在翻译单元顶部时编译时，不能用于启用此行为。在这种情况下，需要使用编译标志，或者需要使用编译标志来定义宏。`--default-stream per-thread``CUDA_API_PER_THREAD_DEFAULT_STREAM``-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1**



对于使用编译标志编译的代码，默认流是一个称为*NULL 流的*特殊流，每个设备都有一个用于所有主机线程的 NULL 流。NULL 流很特殊，因为它会导致隐式同步，如[隐式同步](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#implicit-synchronization)中所述。`--default-stream legacy --default-stream`对于在未指定编译标志的情况下编译的代码，假定为默认值。`--default-stream legacy`



##### 显式同步

有多种方法可以显式地相互同步流。

- `cudaDeviceSynchronize()`等待直到所有主机线程的所有流中的所有先前命令都完成。
- `cudaStreamSynchronize()`将流作为参数并等待，直到给定流中的所有先前命令都完成。它可用于将主机与特定流同步，从而允许其他流继续在设备上执行。
- `cudaStreamWaitEvent()`将流和事件作为参数（有关事件的描述，请参阅[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)），并在调用后将所有命令添加到给定流，以`cudaStreamWaitEvent()`延迟其执行，直到给定事件完成。
- `cudaStreamQuery()`为应用程序提供了一种方法来了解流中所有前面的命令是否已完成。



##### 隐式同步

如果主机线程在来自不同流的两个命令之间发出以下任一操作，则它们不能同时运行：

- 页锁定主机内存分配，
- 设备内存分配，
- 设备内存集，
- 两个地址之间的内存复制到同一设备内存，
- 任何 CUDA 命令到 NULL 流，
- [计算能力 7.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-7-x)中描述的 L1/共享内存配置之间的切换。

需要依赖性检查的操作包括与正在检查的启动相同的流中的任何其他命令以及对该`cudaStreamQuery()`流的任何调用。因此，应用程序应遵循以下准则来提高并发内核执行的潜力：

- 所有独立操作应在相关操作之前发出，
- 任何类型的同步都应尽可能延迟。



##### 重叠行为

两个流之间的执行重叠量取决于向每个流发出命令的顺序以及设备是否支持数据传输和内核执行的重叠（请参阅[数据传输和内核执行的重叠](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#overlap-of-data-transfer-and-kernel-execution)）、并发内核执行（请参阅[并发内核执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-kernel-execution)）和/或并发数据传输（请参阅[并发数据传输](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-data-transfers)）。

例如，在**不支持并发数据传输的设备上，[创建和销毁](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction-streams)代码示例的两个流根本不重叠**，因为从主机到设备的内存复制是在从设备的内存复制之后发出到stream[1]的to host 被发送到stream[0]，因此只有在发送到stream[0] 的从设备到主机的内存复制完成后才能开始。如果代码按以下方式重写（并假设设备支持数据传输和内核执行的重叠）

```c++
for(int i=0;i<2;++i){
  cudaMemcpyAsync(inputDevPtr+i*size,hostPtr+i*size,size,cudaMemcpyHostToDevice,stream[i]);
}

for(int i=0;i<2;++i){
Mykernel<<<100,512,0,stream[i]>>>(outputDevPtr+i*size,inputDevPtr+i*size,size);
}

for(int i=0;i<2;++i){
cudaMemcpyAsync(hostPtr+i*size,outputDevPtr+i*size,size,cudaMemcpyDeviceToHost,stream[i]);
}
```

那么从主机到设备的内存复制发布到stream[1]与发布到stream[0]的内核启动重叠。



在支持并发数据传输的设备上，[创建和销毁](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creation-and-destruction-streams)代码示例的两个流确实重叠：发送到流[1]的从主机到设备的内存复制与发送到流[0]的从设备到主机的内存复制重叠]，甚至将内核启动发布到stream[0]（假设设备支持数据传输和内核执行的重叠）。



##### 主机函数（回调）

运行时提供了一种通过`cudaLaunchHostFunc()`. 一旦回调完成之前向流发出的所有命令都将在主机上执行所提供的函数。

以下代码示例`MyCallback`在向<u>每个流发出主机到设备内存复制、内核启动和设备到主机内存复制后，将主机函数添加到两个流中的每一个。每次设备到主机的内存复制完成后，该函数将开始在主机上执行。</u>

```c++
void CUDART_CB MyCallback(void *data){
  printf("Inside callback %d\n",(size_t)data);
}
for (size_t i =0 ;i<2;++i){
  cudaMemcpyAsync(devPtrIn[i],hostPtr[i],size,cudaMemcpyHostToDevice,stream[i]);
  
MyKernel<<<100,512,0,stream[i]>>>(devPtrOut[i],devPtrIn[i],size);
 
cudaMemcpyAsync(hostPtr[i],devPtrOut[i],size,cudaMemcpyDevicetoHost,stream[i]);
cudaLaunchHostFunc(stream[i],MyCallback,(void*)i);
}
```



在主机函数之后在流中发出的命令在该函数完成之前不会开始执行。

排队到流中的主机函数不得进行 CUDA API 调用（直接或间接），因为如果进行此类调用导致死锁，它最终可能会自行等待。





##### 流优先级

流的相对优先级可以在创建时使用指定`cudaStreamCreateWithPriority()`。可以使用该函数获得允许的优先级范围，排序为[最高优先级，最低优先级] `cudaDeviceGetStreamPriorityRange()`。在运行时，高优先级流中的待处理工作优先于低优先级流中的待处理工作。

以下代码示例获取当前设备允许的优先级范围，并创建具有最高和最低可用优先级的流。

```c++
// get the range of stream priorities for this device 
int priority_high,prority_low;
cudaDeviceGetStreamProiorityRange(&priority_low,&priority_high);
// create stream with highest and lowest available priorities
cudaStream_t st_high,st_low;
cudaStreamCreateWithPriority(&st_high,cudaStreamNonBlocking,priority_high);
cudaStreamCreateWithPriority(&st_low,cudaStreamNonBlocking,priority_low);
```



#### 程序化相关启动和同步

程序*化依赖启动*机制允许依赖的*辅助内核在同一 CUDA 流中依赖的主*内核完成执行之前启动。从计算能力 9.0 的设备开始可用，当*辅助*内核可以完成不依赖于*主*内核结果的重要工作时，该技术可以提供性能优势。



##### 背景

CUDA 应用程序通过在 GPU 上启动和执行多个内核来利用 GPU。[典型的 GPU 活动时间线如图 10](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#gpu-activity)所示。

<img src = 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/gpu-activity.png'>



这里，`secondary_kernel`是在`primary_kernel`执行完成后启动的。串行执行通常是必要的，因为`secondary_kernel`取决于`primary_kernel`. 如果`secondary_kernel`没有依赖性，则可以使用[CUDA 流](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)`primary_kernel`同时启动它们。即使依赖于，也存在并发执行的一些潜力。例如，几乎所有内核都有某种*前导码*部分，在此期间执行诸如清零缓冲区或加载常量值之类的任务。`secondary_kernel``primary_kernel`

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/secondary-kernel-preamble.png'>

上图`secondary_kernel`演示了可以同时执行而不影响应用程序的部分。请注意，<u>并发启动还允许我们隐藏`secondary_kernel`执行后的启动延迟`primary_kernel`。</u>

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/preamble-overlap.png">

*编程相关启动*引入了对 CUDA 内核启动 API 的更改，如下节所述。这些 API 至少需要 9.0 的计算能力才能提供重叠执行。

##### 接口说明

在程序化相关启动中，主内核和辅助内核在同一 CUDA 流中启动。当主内核`cudaTriggerProgrammaticLaunchCompletion`准备好启动辅助内核时，应使用所有线程块执行。辅助内核必须使用可扩展启动 API 启动，如图所示。



> Programmatic_Dependent_Launch_api.cu

当使用该`cudaLaunchAttributeProgrammaticStreamSerialization`属性启动辅助内核时，CUDA 驱动程序可以安全地提前启动辅助内核，而不是在启动辅助内核之前等待主内核的完成和内存刷新。

当所有主线程块已启动并执行时，CUDA 驱动程序可以启动辅助内核 `cudaTriggerProgrammaticLaunchCompletion`。如果主内核不执行触发器，则它会在主内核中的所有线程块退出后隐式发生。

在任何一种情况下，辅助线程块都可能在主内核写入的数据可见之前启动。因此，当辅助内核配置为*“程序化依赖启动”*时，它必须始终使用`cudaGridDependencySynchronize` 或其他方式来验证来自主内核的结果数据是否可用。

请注意，这些方法为主内核和辅助内核提供了并发执行的机会，但是这种行为是机会主义的，并不能保证导致并发内核执行。以这种方式依赖并发执行是不安全的，并且可能导致死锁。



##### 在CUDA图表中使用

编程相关启动可通过[流捕获](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture)或直接通过[边缘数据在](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#edge-data)[CUDA 图形](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)中使用。要在具有边缘数据的 CUDA 图中对此功能进行编程，请在连接两个内核节点的边缘上使用的值。这种边缘类型使得上游内核对下游内核可见。此类型必须与或 的传出端口一起使用。`cudaGraphDependencyType``cudaGraphDependencyTypeProgrammatic``cudaGridDependencySynchronize()``cudaGraphKern`

流捕获的结果图等效如下

| 源代码                                                       | 生成的图形边缘                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization; attribute.val.programmaticStreamSerializationAllowed = 1; | cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; |
| cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 0; | cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortProgrammatic; |
| cudaLaunchAttribute attribute; attribute.id = cudaLaunchAttributeProgrammaticEvent; attribute.val.programmaticEvent.triggerAtBlockStart = 1; | cudaGraphEdgeData edgeData; edgeData.type = cudaGraphDependencyTypeProgrammatic; edgeData.from_port = cudaGraphKernelNodePortLaunchCompletion; |





#### CUDA图形

CUDA Graphs 提出了 CUDA 中工作提交的新模型。图是一系列通过依赖关系连接的操作，例如内核启动，这些操作是与其执行分开定义的。这允许图形被定义一次，然后重复启动。将图的定义与其执行分离可以实现许多优化：首先，与流相比，CPU 启动成本降低，因为大部分设置都是提前完成的；其次，将整个工作流程呈现给 CUDA 可以实现优化，而这对于流的分段工作提交机制来说是不可能实现的。

要了解图形可能进行的优化，请考虑流中发生的情况：当您将内核放入流中时，主机驱动程序会执行一系列操作，为在 GPU 上执行内核做好准备。这些设置和启动内核所必需的操作是必须为每个发布的内核支付的开销成本。对于执行时间较短的 GPU 内核，此开销成本可能占整个端到端执行时间的很大一部分。

使用图的工作提交分为三个不同的阶段：定义、实例化和执行。

- 在定义阶段，程序创建图中操作的描述以及它们之间的依赖关系。
- 实例化会拍摄图形模板的快照，对其进行验证，并执行大部分设置和初始化工作，目的是最大限度地减少启动时需要完成的工作。生成的实例称为*可执行图。*
- 可执行图可以启动到流中，类似于任何其他 CUDA 工作。它可以启动任意多次，而无需重复实例化。



##### 图结构

一个操作形成图中的一个节点。操作之间的依赖关系是边。这些依赖关系限制了操作的执行顺序。

一旦操作所依赖的节点完成，就可以随时安排操作。调度由 CUDA 系统决定。

###### 节点类型

图形节点可以是以下之一：

- 核心
- CPU函数调用
- 内存复制
- 内存设置
- 空节点
- 等待[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)
- 记录[事件](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#events)
- [向外部信号量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#external-resource-interoperability)发出信号
- 等待[外部信号量](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#external-resource-interoperability)
- [条件节点](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#conditional-graph-nodes)
- 子图：执行单独的嵌套图，如下图所示。

<img src = "https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/child-graph.png">



###### 边缘数据

CUDA 12.3在 CUDA 图形中引入了边缘数据。边缘数据修改由边缘指定的依赖项，由三部分组成: 输出端口、输入端口和类型。输出端口指定触发关联边缘的时间。传入端口指定节点的哪一部分依赖于关联的边。类型修改端点之间的关系。



端口值特定于节点类型和方向，边缘类型可以限制为特定的节点类型。在所有情况下，初始化为零的边缘数据表示默认行为。输出端口0等待整个任务，输入端口0阻塞整个任务，边缘类型0与具有内存同步行为的完全依赖关联。



边缘数据可选地通过与关联节点的并行数组在各种图形 API 中指定。如果省略它作为输入参数，则使用初始化为零的数据。如果它作为一个输出(查询)参数被忽略，那么如果被忽略的边缘数据都是零初始化的，那么 API 就会接受这个参数，如果调用会丢弃信息，那么 API 就会返回` cudaErrorLossyQuery`。



边缘数据也可以在一些流捕获 API 中获 `cudaStreamBeginCaptureToGraph ()`、 `cudaStreamGetCaptureInfo ()`和 `cudaStreamUpdateCaptureDependency ()`。在这些情况下，还没有下游节点。数据与一个悬空边(半边)相关联，悬空边要么连接到未来捕获的节点，要么在流捕获结束时丢弃。注意，有些边缘类型不会等待上游节点完全完成。当考虑流捕获是否已完全重新连接到原始流时，这些边将被忽略，并且不能在捕获结束时丢弃。请参见使用流捕获创建图。

目前，没有节点类型定义额外的传入端口，只有内核节点定义额外的传出端口。有一种非默认的依赖类型 `cudaGraphDependencyTypeProgramming`，它支持在两个内核节点之间启动程序相关启动。





##### 使用图的api来创建图

图可以通过两种机制创建: 显式 API 和流捕获。下面是创建和执行下图的示例。

<img src ="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/create-a-graph.png">



> create_graph.cu



##### 创建图通过流捕获

流捕获提供了一种从现有的基于流的 API 创建图的机制。将工作启动到流(包括现有代码)中的代码段可以用调用 udaStreamBeginCapture ()和 udaStreamEndCapture ()来括号。请看下面。

```
//3,2,8,7,3 Creating a Graph Using Stream Capture
cudaGraph_t graph;
cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);
```



调用 cudaStreamBeginCapture ()将流置于捕获模式。当捕获流时，发射到流中的工作不会排队等待执行。相反，它被附加到一个正在逐步构建的内部图中。然后通过调用 cudaStreamEndCapture ()返回该图，该函数也结束了流的捕获模式。通过流捕获积极构造的图称为捕获图。



流捕获可以用于任何 CUDA 流，除了 cudaStreamLegacy (“ NULL 流”)。请注意，它可以在 cudaStreamPerThread 上使用。如果程序正在使用遗留流，那么可以重新定义流0，使其成为每个线程的流，而不需要进行功能更改。请参见默认流。



是否捕获流可以使用 cudaStreamIsCapuring ()查询。

工作可以使用 cudaStreamBeginCaptureToGraph ()捕获到现有的图。工作不是捕获到内部图，而是捕获到用户提供的图。



###### 跨流依赖项和事件

流捕获可以处理用 cudaEventRecord ()和 cudaStreamWaitEvent ()表示的跨流依赖，前提是正在等待的事件被记录到同一个捕获图中。



当事件记录在处于捕获模式的流中时，将导致捕获事件。捕获的事件表示捕获图中的一组节点。



当一个捕获的事件被一个流等待时，如果它还没有被捕获，那么它将该流置于捕获模式中，并且流中的下一个项目将对捕获的事件中的节点具有附加的依赖关系。然后将这两个流捕获到同一个捕获图中。



当流捕获中存在跨流依赖关系时，必须仍然在调用 cudaStreamBeginCapture ()的同一流中调用 cudaStreamEndCapture () ; 这是原始流。由于基于事件的依赖关系，被捕获到同一捕获图的任何其他流也必须被连接回原始流。这一点如下图所示。在 cudaStreamEndCapture ()时，所有被捕获到相同捕获图的流都将脱离捕获模式。未能重新加入原始流将导致整个捕获操作失败。

```c++
// stream1 is the origin stream
cudaStreamBeginCapture(stream1);
kernel_A<<<...,stream1>>>(...);
// fork into stream2 
cudaEventRecord(event1,stream1);
cudaStreamWaitEvent(stream2,event1);

kernel_B<<<...,stream1>>>(...);
kernel_C<<<...,stream1>>>(...);

// join stream2 back to origin stream
cudaEventRecord(event2,stream2);
cudaStreamWaitEvents(stream1,event2);
kernel_D<<<...,stream1>>>(...);

//end capture in the origin stream
cudaStreamEndCapture(stream1,&graph);
// stream1 and stream2 no longer in capture mode 
```



**笔记**

**当一个流脱离捕获模式时，流中的下一个未捕获项(如果有的话)仍然依赖于最近的前一个未捕获项，尽管中间项已被删除。**



###### 禁止和未经处理的操作

> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#prohibited-and-unhandled-operations



同步或查询正在捕获的流或捕获的事件的执行状态是无效的，因为它们不表示计划执行的项。当任何相关流处于捕获模式时，查询包含活动流捕获(如设备或上下文句柄)的更广句柄的执行状态或同步该句柄也是无效的。



当捕获同一上下文中的任何流时，如果没有使用 `cudaStreamNonBlock `创建该流，则任何尝试使用遗留流的做法都是无效的。这是因为遗留流句柄在任何时候都包含这些其他流; 排队到遗留流将创建对被捕获的流的依赖关系，而查询或同步它将查询或同步被捕获的流。



因此，在这种情况下调用同步 API 也是无效的。同步 API (比如 `cudaMemcpy ()`将工作放入遗留流中，并在返回之前对其进行同步。



**笔记**

**一般规则，当依赖关系将与未捕获的东西捕获的东西连接并为执行起来时，Cuda更喜欢返回错误而不是忽略依赖关系。将流置于捕获模式或之外的例外；这会在模式转换之前和之后的立即添加到流中的项目之间的依赖关系。**



通过等待正在捕获的流中捕获的事件并与事件不同的捕获图关联，可以合并两个单独的捕获图。从未指定`cudaeventwaitexternal`标志的情况下等待正在捕获的流中的非捕捉事件是无效的。

当前，图形中没有支持少数涉及异步操作进入流中的API，如果用正在捕获的流（例如`Cudastreamattachmemamync（）`调用流中，将返回错误。



###### 无效

當在流捕獲期間嘗試無效操作時，任何關聯的捕獲圖都會*失效*。當捕獲圖無效時，進一步使用任何正在捕獲的流或捕獲與該圖關聯的事件都是無效的，並將返回錯誤，直到流捕獲以 結束`cudaStreamEndCapture()`。此呼叫將使關聯的流退出捕獲模式，但也會傳回錯誤值和 NULL 圖。

##### CUDA使用者对象

CUDA 使用者物件可用於協助管理 CUDA 中非同步工作所使用的資源的生命週期。特別是，此功能對於[CUDA 圖形](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)和[串流捕獲](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture)非常有用。

各種資源管理方案與 CUDA 圖不相容。例如，考慮基於事件的池或同步創建、非同步銷毀方案。

```c++
//library API with pool allocation
void libraryWork(cudaStream_t stream){
    auto &resource = pool.claimTemporaryResource();
    resource.waitOnReadyEventInStream(stream);
    launchWork(stream,resource);
    resource.recordReadyEvent(stream);
}

// library api with asynchoronous resource deletion
void libraryWork(cudaStream_t stream){
    Resource *resource = new Resource(...);
    launchWork(stream,resource);
    cudaStreamAddCallback(
    stream,[](cudaStream_T,cudaError_t,void * resource){
        delete static_cast<Resource *>(resource);
    },resource,0);
    // error handlding considerations not shown 
}
```



這些方案對於 CUDA 圖來說很困難，因為資源的非固定指標或句柄需要間接或圖更新，每次提交工作時都需要同步 CPU 程式碼。如果這些注意事項對庫的呼叫者隱藏，並且由於在捕獲期間使用了不允許的 API，它們也無法與流捕獲一起使用。存在多種解決方案，例如將資源公開給呼叫者。CUDA 使用者物件提供了另一種方法。

CUDA 使用者物件將使用者指定的析構函數回呼與內部引用計數相關聯，類似於 C++ `shared_ptr`。引用可能由 CPU 上的使用者代碼和 CUDA 圖擁有。請注意，對於使用者擁有的引用，與 C++ 智慧指標不同，沒有表示引用的物件；用戶必須手動追蹤用戶擁有的引用。典型的用例是在建立使用者物件後立即將唯一使用者擁有的參考移至 CUDA 圖形。

當引用與 CUDA 圖關聯時，CUDA 將自動管理圖操作。克隆保留來源`cudaGraph_t`擁有的每個引用的副本`cudaGraph_t`，具有相同的多重性。實例化`cudaGraphExec_t`保留來源中每個引用的副本`cudaGraph_t`。當 a 在`cudaGraphExec_t`沒有同步的情況下被銷毀時，引用將被保留，直到執行完成。

使用范例

> cuda_obj.cu





子图节点中的图所拥有的引用与子图相关联，而不是与父图相关联。如果更新或删除子图，引用也会相应更改。如果使用 或 更新可执行图或子图`cudaGraphExecUpdate`，`cudaGraphExecChildGraphNodeSetParams`则会克隆新源图中的引用并替换目标图中的引用。在任一情况下，如果先前的启动未同步，则将保留将释放的任何引用，直到启动完成执行。

目前没有一种机制可以通过 CUDA API 等待用户对象析构函数。用户可以从析构函数代码手动发出同步对象信号。此外，从析构函数中调用 CUDA API 是不合法的，类似于`cudaLaunchHostFunc`. 这是为了避免阻塞 CUDA 内部共享线程并阻止前进。如果依赖关系是单向的并且执行调用的线程不能阻止 CUDA 工作的向前进展，则向另一个线程发出信号以执行 API 调用是合法的。

用户对象是使用 来创建的`cudaUserObjectCreate`，这是浏览相关 API 的一个很好的起点。





##### 更新实例化图

<u>使用图的工作提交分为三个不同的阶段：定义、实例化和执行。在工作流程不变的情况下，定义和实例化的开销可以在多次执行中分摊，并且图形比流具有明显的优势</u>。



图表是工作流程的快照，包括内核、参数和依赖项，以便尽可能快速有效地重放它。在工作流程发生变化的情况下，图表就会过时并且必须进行修改。对图结构（例如拓扑或节点类型）的重大更改将需要重新实例化源图，因为必须重新应用各种与拓扑相关的优化技术。



<u>重复实例化的成本会降低图执行带来的整体性能优势，但通常只有节点参数（例如内核参数和`cudaMemcpy`地址）发生变化，而图拓扑保持不变。</u>对于这种情况，CUDA 提供了一种称为“图形更新”的轻量级机制，它允许就地修改某些节点参数，而无需重建整个图形。这比重新实例化要高效得多。





更新将在下次启动图表时生效，因此它们不会影响之前的图表启动，即使它们在更新时正在运行。图表可以重复更新和重新启动，因此多个更新/启动可以在流上排队。

CUDA提供了两种更新实例化图参数的机制：全图更新和单个节点更新。整个图更新允许用户提供拓扑相同的`cudaGraph_t`对象，其节点包含更新的参数。单个节点更新允许用户显式更新单个节点的参数。`cudaGraph_t`当更新大量节点时，或者当调用者未知图拓扑时（即，由库调用的流捕获产生的图），使用更新更方便。当更改数量较小且用户拥有需要更新的节点的句柄时，首选使用单个节点更新。单个节点更新会跳过未更改节点的拓扑检查和比较，因此在许多情况下会更有效。

CUDA 还提供了一种启用和禁用各个节点而不影响其当前参数的机制。

以下部分更详细地解释了每种方法。





###### 图表更新限制

内核节点

- 函数所属的上下文不能更改
- 原本不使用CUDA动态并行功能的节点无法更新为使用CUDA动态并行功能的节点。

`cudaMemset`和`cudaMemcpy`节点

- 分配/映射操作数的CUDA设备无法更改
- 源/目标内存必须从原始源/目标内存相同的上下文中分配
- 只能更改一堆`cudaMemset`和`cudaMemcpy`

其他`memcpy`节点限制

- 不支持更改源或目标内存类型（即`cudaPitchedPtr`、`cudaArray_t`等）或传输类型（即）。`cudaMemcpyKind`

外部信号量等待节点或记录节点

- 不支持更改信号量的数量

条件节点

- 图之间句柄创建和分配顺序必须匹配
- 不支持更改节点参数（即条件，节点上下文中的图形数量等）
- 更改条件体图中节点的参数必须遵守上述规则

对主机节点、事件记录节点或事件等待节点的更新没有限制。



###### 全图更新

`cudaGraphExecUpdate()`允许使用拓扑相同的图（“更新”图）中的参数更新实例化图（“原始图”）。更新图的拓扑必须与用于实例化 的原始图相同`cudaGraphExec_t`。此外，指定依赖项的顺序必须匹配。最后，CUDA 需要对汇聚节点（没有依赖关系的节点）进行一致的排序。CUDA依赖于特定api调用的顺序来实现一致的sink节点排序。

更明确地说，遵循以下规则将导致`cudaGraphExecUpdate()`原始图中的节点和更新图中的节点确定性地配对：

1. 对于任何捕获流，对该流进行操作的 API 调用必须以相同的顺序进行，包括事件等待和其他与节点创建不直接对应的 API 调用。
2. 直接操作给定图节点的传入边的 API 调用（包括捕获流 API、节点添加 API 和边添加/删除 API）必须以相同的顺序进行。此外，当在数组中指定这些 API 的依赖项时，在这些数组内指定依赖项的顺序必须匹配。
3. 接收器节点的顺序必须一致。汇节点是调用时最终图中没有依赖节点/传出边的节点`cudaGraphExecUpdate()`。以下操作会影响接收器节点排序（如果存在）并且必须（作为组合集）以相同的顺序进行：
   - 节点添加 API 产生接收器节点。
   - 边缘移除导致节点成为汇聚节点。
   - `cudaStreamUpdateCaptureDependencies()`，如果它从捕获流的依赖集中删除接收器节点。
   - `cudaStreamEndCapture()`。

以下示例展示了如何使用 API 来更新实例化图：

> update_graph.cu



`cudaGraph_t`典型的工作流程是使用流捕获或图形 API创建初始值。然后实例`cudaGraph_t`化并正常启动。初始启动后，`cudaGraph_t`将使用与初始图相同的方法创建一个新图并`cudaGraphExecUpdate()`进行调用。如果图形更新成功（如上例中的参数所示） ，则启动`updateResult`更新。`cudaGraphExec_t`如果由于任何原因更新失败，则会调用`cudaGraphExecDestroy()`和`cudaGraphInstantiate()`来销毁原始更新`cudaGraphExec_t`并实例化一个新更新。

也可以`cudaGraph_t`直接更新节点（即使用`cudaGraphKernelNodeSetParams()`）并随后更新`cudaGraphExec_t`，但是使用下一节中介绍的显式节点更新 API 会更有效。

条件句柄标志和默认值作为图形更新的一部分进行更新。

请参阅[Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH)了解有关使用和当前限制的更多信息。





###### 单个节点更新

实例化的图节点参数可以直接更新。这消除了实例化的开销以及创建新的`cudaGraph_t`. 如果需要更新的节点数量相对于图中的节点总数而言较少，则最好单独更新节点。以下方法可用于更新`cudaGraphExec_t`节点：

- `cudaGraphExecKernelNodeSetParams()`
- `cudaGraphExecMemcpyNodeSetParams()`
- `cudaGraphExecMemsetNodeSetParams()`
- `cudaGraphExecHostNodeSetParams()`
- `cudaGraphExecChildGraphNodeSetParams()`
- `cudaGraphExecEventRecordNodeSetEvent()`
- `cudaGraphExecEventWaitNodeSetEvent()`
- `cudaGraphExecExternalSemaphoresSignalNodeSetParams()`
- `cudaGraphExecExternalSemaphoresWaitNodeSetParams()`



###### 单个节点启用

可以使用 cudaGraphNodeSetEnabled() API 启用或禁用实例化图中的内核、memset 和 memcpy 节点。这允许创建一个图表，其中包含所需功能的超集，可以为每次启动进行自定义。可以使用 cudaGraphNodeGetEnabled() API 查询节点的启用状态。

禁用的节点在功能上等同于空节点，直到重新启用为止。节点参数不受启用/禁用节点的影响。启用状态不受单个节点更新或使用 cudaGraphExecUpdate() 更新整个图的影响。节点禁用时的参数更新将在节点重新启用时生效。

以下方法可用于启用/禁用`cudaGraphExec_t`节点以及查询其状态：

- `cudaGraphNodeSetEnabled()`
- `cudaGraphNodeGetEnabled()`





##### 使用图形API

`cudaGraph_t`对象不是线程安全的。用户有责任确保多个线程不会同时访问同一个`cudaGraph_t`.

A`cudaGraphExec_t`不能与其自身同时运行。a 的启动`cudaGraphExec_t`将在先前启动同一可执行图之后进行。

图形执行在流中完成，以便与其他异步工作进行排序。然而，该流仅用于订购；它不限制图的内部并行性，也不影响图节点的执行位置。





##### 设备图启动

有许多工作流需要在运行时做出依赖于数据的决策，并根据这些决策执行不同的操作。用户可能更愿意在设备上执行此决策过程，而不是将此决策过程卸载到主机（这可能需要从设备进行往返）。为此，CUDA 提供了一种从设备启动图形的机制。

设备图启动提供了一种从设备执行动态控制流的便捷方法，无论是简单的循环还是复杂的设备端工作调度程序。[此功能仅在支持统一寻址](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#unified-virtual-address-space)的系统上可用。

可以从设备启动的图此后将被称为设备图，而不能从设备启动的图将被称为主机图。

<u>设备图可以从主机和设备启动，而主机图只能从主机启动。</u>与主机启动不同，在先前启动的图正在运行时从设备启动设备图将导致错误，返回`cudaErrorInvalidValue`；因此，设备图不能同时从设备启动两次<u>。同时从主机和设备启动设备图将导致未定义的行为</u>。



###### 设备图创建

为了从设备启动图表，必须为设备启动显式实例化它。这是通过将`cudaGraphInstantiateFlagDeviceLaunch`标志传递给`cudaGraphInstantiate()`调用来实现的。与主机图的情况一样，设备图结构在实例化时是固定的，如果不重新实例化就无法更新，并且实例化只能在主机上执行。为了使图能够在设备启动时实例化，它必须遵守各种要求。



**设备图要求**

一般要求：

- 图表的节点必须全部驻留在单个设备上。
- 该图只能包含内核节点、memcpy 节点、memset 节点和子图节点。

内核节点：

- 不允许图中的内核使用 CUDA 动态并行性。
- 只要不使用 MPS，就允许合作发射。

Memcpy 节点：

- 仅允许涉及设备内存和/或固定设备映射主机内存的副本。
- 不允许涉及 CUDA 数组的副本。
- 在实例化时，两个操作数都必须可以从当前设备访问。请注意，复制操作将从图形所在的设备执行，即使它的目标是另一个设备上的内存。

**设备图上传**

为了在设备上启动图表，必须首先将其上传到设备以填充必要的设备资源。这可以通过两种方式之一来实现。

`cudaGraphUpload()`首先，可以通过或通过请求上传作为实例化的一部分来显式上传图表`cudaGraphInstantiateWithParams()`。

或者，可以首先从主机启动图表，主机将在启动过程中隐式执行此上传步骤。

所有三种方法的示例如下：

```c++
//explicit upload after instatntiation 
cudaGraphInstantiate(&deviceGraphExec1, deviceGraph1, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphUpload(deviceGraphExec1, stream);

// Explicit upload as part of instantiation
cudaGraphInstantiateParams instantiateParams = {0};
instantiateParams.flags = cudaGraphInstantiateFlagDeviceLaunch | cudaGraphInstantiateFlagUpload;
instantiateParams.uploadStream = stream;
cudaGraphInstantiateWithParams(&deviceGraphExec2, deviceGraph2, &instantiateParams);

// Implicit upload via host launch
cudaGraphInstantiate(&deviceGraphExec3, deviceGraph3, cudaGraphInstantiateFlagDeviceLaunch);
cudaGraphLaunch(deviceGraphExec3, stream);

```

 **设备图更新**

设备图只能从主机更新，并且必须在可执行图更新后重新上传到设备才能使更改生效。这可以使用上一节中概述的相同方法来实现。与主机图不同，在应用更新时从设备启动设备图将导致未定义的行为。





###### 设备启动

设备图可以通过主机和设备启动`cudaGraphLaunch()`，设备上的签名与主机上的签名相同。设备图通过主机和设备上的相同句柄启动。从设备启动时，设备图必须从另一个图启动。

设备端图启动是按线程进行的，不同线程可能同时发生多个启动，因此用户需要选择一个线程来启动给定图。

**设备启动模式**

与主机启动不同，设备图不能启动到常规 CUDA 流中，只能启动到不同的命名流中，每个流都表示特定的启动模式：

| stream                                  | 启动模式     |
| --------------------------------------- | ------------ |
| `cudaStreamGraphFiredAndForget`         | 即发即忘发射 |
| `cudaStreamGraphTailLaunch`             | 尾部发射     |
| `cudaStreamGraphFireAndForgetAsSibling` | 兄弟姐妹发射 |

**即发即忘发射**

顾名思义，即发即忘启动会立即提交给 GPU，并且独立于启动图运行。在“即发即忘”场景中，启动图是父图，启动图是子图。

<img src= 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-simple.png'>

> FireAndForget.cu



一个图在其执行过程中最多可以有 120 个即发即弃图。此总数会在同一父图的启动之间重置。



**图执行虚拟环境**

为了充分理解设备端同步模型，首先需要了解执行环境的概念。



从设备启动图表时，将其启动到自己的执行环境中。给定图的执行环境封装了图中的所有作品，以及所有生成的火和忘记工作。该图完成执行后以及所有生成的子工作完成后，可以将其视为完整。



下图显示了上一节中的火与验样示例代码将生成的环境封装。

<img src ='https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-environments.png'>

这些环境也是层次结构，因此图形环境可以包括火和忘记发射的多个级别的儿童环境。

<img src= "https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/fire-and-forget-nested-environments.png">

当从主机启动图形时，存在一个流环境，使启动图的执行环境父母。流环境封装了作为整体发射的一部分生成的所有作品。当总体流环境标记为完整时，流启动已完成（即现在可以运行下游依赖性工作）。

<img src= 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/device-graph-stream-environment.png'>



**tail launch** 

与主机上不同，不可能通过传统方法（例如`cudadevicessynchronize（）`或`cudastreamsynchronize（）`（）（）与GPU的设备图同步。相反，为了启用串行工作依赖性，提供了不同的启动模式 - 尾巴启动 - 提供类似的功能。



当将图的环境视为完整时，尾巴发射将执行 - 即，当图形及其所有孩子都完成时。图表完成后，尾部启动列表中下一个图的环境将替换为父母环境的童年。像火和孔的发射一样，图可以具有用于尾部发射的多个图形。

> tail_launch.cu

<img src ="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-simple.png">

由给定图出现的尾巴发射将按照何时被启用。因此，第一个居住的图将首先运行，然后是第二个，依此类推。

<img src ="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-ordering-simple.png">

在尾部发射列表中以前的图被启动之前，由尾部图出现的尾巴发射将执行。这些新的尾巴发射将按照它们被晋升的顺序执行。

<img src="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/tail-launch-ordering-complex.png">

图可以具有多达255个待处理的尾巴发射。





**尾巴自动发射**

设备图可能会出现自身以尾声发射，<u>尽管给定的图只能一次启动一个自我启动</u>。为了查询当前运行的设备图，以便可以重新启动，添加了一个新的设备端功能：

```c
cudaGraphExec_t cudaGetCurrentGraphExec();
```





如果该功能是设备图，则此功能将返回当前运行图的句柄。如果当前执行的内核不是设备图中的节点，则此函数将返回null。

Below is sample code showing usage of this function for a relaunch loop:

> tail_re_launch.cu



**同步发射**

同步发射的发布是Fire-Forget发射的一种变体，其中该图不是作为启动图的执行环境的孩子，而是作为启动图的父母环境的孩子。兄弟姐妹的发布等同于启动图的父母环境中的火灾和武器发射。

<img src ="https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/sibling-launch-simple.png">

> sibling_launch.cu
