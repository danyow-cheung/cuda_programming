### 图形互操作性

可以将OpenGL和Direct3D的一些资源映射到CUDA的地址空间中，以使CUDA能够读取由OpenGL或Direct3D编写的数据，或者使CUDA能够通过OpenGL或Direct3D编写数据供消费。



必须使用OpenGL互操作性和Direct3D互操作性中提到的功能对CUDA进行注册。这些功能将指针返回到类型`CudagraphicsResource`类型的CUDA图形资源。注册资源可能是高空的高空，因此通常只调用一次每个资源。使用`cudagraphicsunregisterresource（）`未注册CUDA图形资源。打算使用资源的每个CUDA上下文都需要单独注册。

一旦将资源注册到CUDA，就可以使用`CudagraphicsMapResources（）`和`Cudagraphicsunmapresources（）`来映射和未映射。可以调用`CudagraphicsResourcesetMapflags（）`来指定CUDA驱动程序可以用来优化资源管理的CUDA驱动程序的用法提示（仅写，只读）。



可以使用`cudagraphicsresourcegegetmappedpointer（）`返回的buffers和`cudagraphicsSubresourcegegetMappaparay（`）返回的设备存储器地址（）返回的设备内存地址，可以从内核中读取或编写映射资源。



通过映射时，通过OpenGL，Direct3D或其他CUDA上下文访问资源会产生未定义的结果。OpenGL互操作性和Direct3D互操作性为每个图形API和一些代码示例提供了细节。SLI互操作性提供了系统处于SLI模式时的具体内容。





#### OpenGL互操作性

可以映射到 CUDA 地址空间的 OpenGL 资源是 OpenGL 缓冲区、纹理和渲染缓冲区对象。



缓冲区对象是使用 注册的`cudaGraphicsGLRegisterBuffer()`。在 CUDA 中，它显示为设备指针，因此可以由内核或通过调用进行读写`cudaMemcpy()`。

使用 注册纹理或渲染缓冲区对象`cudaGraphicsGLRegisterImage()`。在 CUDA 中，它显示为 CUDA 数组。内核可以通过将数组绑定到纹理或表面引用来读取数组。如果资源已使用标志注册，他们还可以通过表面写入函数对其进行写入`cudaGraphicsRegisterFlagsSurfaceLoadStore`。该数组也可以通过调用来读取和写入`cudaMemcpy2D()`。`cudaGraphicsGLRegisterImage()`支持具有 1、2 或 4 个分量以及浮点数内部类型（例如`GL_RGBA_FLOAT32`）、标准化整数（例如）和非标准化整数（例如）的所有纹理格式（请注意，由于非标准化整数格式需要 OpenGL 3.0，它们只能由着色器编写，不能由固定功能管道编写）。`GL_RGBA8, GL_INTENSITY16``GL_RGBA8UI`

共享资源的 OpenGL 上下文对于进行任何 OpenGL 互操作性 API 调用的主机线程来说必须是最新的。



请注意：当 OpenGL 纹理设为无绑定时（例如通过使用`glGetTextureHandle`*/ `glGetImageHandle`* API 请求图像或纹理句柄），它无法在 CUDA 中注册。应用程序需要在请求图像或纹理句柄之前注册纹理以进行互操作。

以下代码示例使用内核动态修改存储在顶点缓冲区对象中的2D `width`x顶点网格：`height`

> opengl.cu

在 Windows 和 Quadro GPU 上，`cudaWGLGetDevice()`可用于检索与返回的句柄关联的 CUDA 设备`wglEnumGpusNV()`。在多 GPU 配置中，Quadro GPU 提供比 GeForce 和 Tesla GPU 更高性能的 OpenGL 互操作性，其中 OpenGL 渲染在 Quadro GPU 上执行，CUDA 计算在系统中的其他 GPU 上执行。



#### Direct3D 互操作性

Direct3D 9Ex、Direct3D 10 和 Direct3D 11 支持 Direct3D 互操作性。

CUDA 上下文只能与满足以下条件的 Direct3D 设备进行互操作： Direct3D 9Ex 设备必须在创建时设置为`DeviceType`并`D3DDEVTYPE_HAL`带有`BehaviorFlags`标志`D3DCREATE_HARDWARE_VERTEXPROCESSING`；`DriverType`Direct3D 10 和 Direct3D 11 设备必须在设置为 的情况下创建`D3D_DRIVER_TYPE_HARDWARE`。

可以映射到 CUDA 地址空间的 Direct3D 资源是 Direct3D 缓冲区、纹理和表面。`cudaGraphicsD3D9RegisterResource()`这些资源使用、`cudaGraphicsD3D10RegisterResource()`和进行注册`cudaGraphicsD3D11RegisterResource()`。

以下代码示例使用内核动态修改存储在顶点缓冲区对象中的2D `width`x顶点网格。`height`

> ##### Direct3D 10 版本
>
> direct3d_10.cu





#### SLI互操作性

在具有多个 GPU 的系统中，所有支持 CUDA 的 GPU 都可以作为单独的设备通过 CUDA 驱动程序和运行时进行访问。然而，当系统处于 SLI 模式时，有如下所述的特殊注意事项。

首先，一个 GPU 上的一个 CUDA 设备中的分配将消耗其他 GPU 上的内存，这些 GPU 是 Direct3D 或 OpenGL 设备的 SLI 配置的一部分。因此，分配可能会比预期更早失败。

其次，应用程序应创建多个 CUDA 上下文，每个上下文对应 SLI 配置中的每个 GPU。虽然这不是严格要求，但它避免了设备之间不必要的数据传输。应用程序可以使用`cudaD3D[9|10|11]GetDevices()`for Direct3D 和`cudaGLGetDevices()`for OpenGL 调用集来识别在当前帧和下一帧中执行渲染的设备的 CUDA 设备句柄。鉴于此信息，应用程序通常会选择适当的设备，并将 Direct3D 或 OpenGL 资源映射到由`cudaD3D[9|10|11]GetDevices()`或`cudaGLGetDevices()`当`deviceList`参数设置为`cudaD3D[9|10|11]DeviceListCurrentFrame`或时返回的 CUDA 设备`cudaGLDeviceListCurrentFrame`。

请注意，从注册的设备返回的资源`cudaGraphicsD9D[9|10|11]RegisterResource`必须`cudaGraphicsGLRegister[Buffer|Image]`仅在注册发生的设备上使用。因此，在 SLI 配置中，当在不同 CUDA 设备上计算不同帧的数据时，有必要分别注册每个帧的资源。

有关 CUDA 运行时如何分别与 Direct3D 和OpenGL[互操作的](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#opengl-interoperability)详细信息，请参阅Direct3D[互操作性和 OpenGL 互操作性。](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#direct3d-interoperability)



### 外部资源互操作性

外部资源互操作性允许 CUDA 导入由其他 API 显式导出的某些资源。这些对象通常由其他 API 使用操作系统本机句柄导出，例如 Linux 上的文件描述符或 Windows 上的 NT 句柄。它们还可以使用其他统一接口（例如 NVIDIA 软件通信接口）导出。可以导入两种类型的资源：内存对象和同步对象。

内存对象可以使用 导入到 CUDA 中`cudaImportExternalMemory()`。`cudaExternalMemoryGetMappedBuffer()`可以使用通过 映射到内存对象的设备指针或通过 映射的 CUDA mipmap 数组从内核内部访问导入的内存对象`cudaExternalMemoryGetMappedMipmappedArray()`。根据存储器对象的类型，可以在单个存储器对象上设置多个映射。映射必须与导出 API 中的映射设置匹配。任何不匹配的映射都会导致未定义的行为。导入的内存对象必须使用 释放`cudaDestroyExternalMemory()`。释放内存对象不会释放到该对象的任何映射。因此，映射到该对象的任何设备指针必须使用显式释放`cudaFree()`，并且映射到该对象的任何 CUDA mipmap 数组必须使用显式释放`cudaFreeMipmappedArray()`。在对象被销毁后访问对象的映射是非法的。

同步对象可以使用 导入到 CUDA 中`cudaImportExternalSemaphore()`。然后可以使用 向导入的同步对象发出信号`cudaSignalExternalSemaphoresAsync()`并等待`cudaWaitExternalSemaphoresAsync()`。在发出相应信号之前发出等待是非法的。此外，根据导入的同步对象的类型，可能会对如何发出信号和等待它们施加额外的约束，如后续部分所述。导入的信号量对象必须使用 释放`cudaDestroyExternalSemaphore()`。所有未完成的信号和等待必须在信号量对象被销毁之前完成



#### Vulkan 互操作性

##### 匹配设备UUID

导入由 Vulkan 导出的内存和同步对象时，必须将它们导入并映射到创建它们的同一设备上。可以通过将 CUDA 设备的 UUID 与 Vulkan 物理设备的 UUID 进行比较来确定与创建对象的 Vulkan 物理设备对应的 CUDA 设备，如以下代码示例所示。请注意，Vulkan 物理设备不应属于包含多个 Vulkan 物理设备的设备组。返回的包含给定 Vulkan 物理设备的设备组的`vkEnumeratePhysicalDeviceGroups`物理设备计数必须为 1。

```  
```

