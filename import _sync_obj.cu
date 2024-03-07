cudaExternalSemaphore_t importVulkanSemaphoreObjectFromFileDescriptor(int fd){
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoeHeadleDesc desc = {};

    memset(&desc,0,sizeof(desc));
    desc.type = cudaExternalSemaphoeHandleTypeOpaqueFd;

    desc.handle.fd = fd;
    cudaImportExternalSemaphore(&extSem,&desc);

    return extSem;
}

// 使用 VK _ EXTERNAL _ SEMAPHORE _ HANDLE _ TYPE _ OPAQUE _ WIN32 _ BIT 导出的 Vulkan 信号量对象
// 可以使用与该对象关联的 NT 句柄导入 CUDA，
// 如下所示。请注意，CUDA 并不承担 NT 句柄的所有权，当不再需要这个句柄时，关闭它是应用程序的责任。
// NT 句柄持有对资源的引用，因此必须在释放基础信号量之前显式地释放它。


cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNTHandle(HANDLE handle){
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc,0,sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;

    cudaImportExternalSemaphore(&extMem,&desc);

    //input parameters handle shoudld be closed if it's not neeed anymore 
    CloseHandle(handle);
    return extSem;
}

// 使用 VK _ EXTERNAL _ SEMAPHORE _ HANDLE _ TYPE _ OPAQUE _ WIN32 _ BIT 
// 导出的 Vulkan 信号量对象也可以使用命名句柄导入，如下所示。
cudaExternalSemaphore_t importVulkanSemaphoreObjectFromNamedNTHandle(LPCWSTR name){
    cudaExternalSemaphore_t extSem = NULL;
    cudaExteranlSemaphoreHanldeDesc desc = {};

    memset(&desc,0,sizeof(desc));
    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    desc.handle.win32.name = (void *)name;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}


// A Vulkan semaphore object exported using VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT can be imported into CUDA using the globally shared D3DKMT handle associated with that object as shown below. Since a globally shared D3DKMT handle does not hold a reference to the underlying semaphore it is automatically destroyed when all other references to the resource are destroyed.


cudaExternalSemaphore_t importVulkanSemaphoreObjectFromKMTHandle(HANDLE handle) {
    cudaExternalSemaphore_t extSem = NULL;
    cudaExternalSemaphoreHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    desc.handle.win32.handle = (void *)handle;

    cudaImportExternalSemaphore(&extSem, &desc);

    return extSem;
}
