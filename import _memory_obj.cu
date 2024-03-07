cudaExternalMemory_t importVulkanMemoryObjectFromFileDescriptor(int fd,unsigned long long size,bool isDedicated){
	cudaExternalMemory_t exMem = NULL:
	cudaExternalMemoryHandleDesc desc = {};
	
	memset(&desc,0,sizeof(desc));
	desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
	desc.handle.fd = fd;
	desc.size = size;
	if (isDedicated){
	desc.flags |= cudaExternalMemoryDediacated;
	}
	cudaImportExternalMemory(&extMem,&desc);
	
	return extMem;
}

cudaExternalMemory_t importVulkanMemoryObjectFromHandle(HANDLE handle,unsigned long long size,bool isDedicated){
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc,0,sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaquWin32;
    desc.handle.win32.handle = handle;
    desc.size = size;
    if (isDedicated){
        desc.flags |= cudaExternalMemoryDediacated;
    }

    cudaImportExternalMemory(&extMem,&desc);

    CloseHandle(handle);
    return extMem;
}


// 使用导出的 Vulkan 内存对象VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
//也可以使用命名句柄导入（如果存在），如下所示。
cudaExternalMemory_t importVulkanMemoryObjectFromNamedNTHandle(LPCWSTR name,unsigned long long  size ,bool isDedicated){
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc  desc ={};

    memset(&desc,0,sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeOpaquWin32;
    desc.size = size;
    if (isDedicated){

        desc.flags |= cudaExternalMemoryDediacated;
    }
    cudaImportExternalMemory(&extMem,&desc);
}

// 使用 VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT 
// 导出的 Vulkan 内存对象可以使用与该对象关联的全局共享 D3DKMT 句柄导入到 CUDA 中，
// 如下所示。由于全局共享的 D3DKMT 句柄不保存对底层内存的引用，因此当对该资源的所有其他引用被销毁时，它会自动销毁。

cudaExternalMemory_t importVulkanMemoryObjectFromKMTHandle(HANDLE handle,unsigned long long size,bool isDedicated){
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};
    memset(&desc,0,sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeOpaquWin32Kmt;
    desc.handle.win32.handle = (void*)handle;
    desc.size = size;
    if (isDedicated){
        desc.flags |= cudaExternalMemoryDediacated;
    }
    cudaImportExternalMemory(&extMem,&desc);
    return extMem;
}
