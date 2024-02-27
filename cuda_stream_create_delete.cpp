// 流是通过创建流对象并将其指定为一系列内核启动和主机<->设备内存副本的流参数来定义的。
// 以下代码示例创建两个流并分配一个页锁定内存hostPtr数组。
cudaStream_t stream[2];
for(int i = 0;i<2;++i)cudaStreamCreate(&stream[i]);

float* hostPtr;
cudaMalloclocHost(&hostPtr,2*size);
// 每个流都由以下代码示例定义为从主机到设备的一次内存复制、一次内核启动以及从设备到主机的一次内存复制的序列：
for (int i = 0;i<2;++i){
    cudaMemcpyAsync(intputDevPtr+i*size,hostPtr+i*size,size,cudaMemcpyHostToDevice,stream[i]);
    MyKerenl<<<100,512,0,stream[i]>>> (outputDevPtr+i*size,intputDevPtr+i*size,size);
    cudaMemecpyAsync(hostPtr+i*size,outputDevPtr + i * size,
                    size, cudaMemcpyDeviceToHost, stream[i]);
}



// 每个流将其输入数组的部分复制hostPtr到inputDevPtr设备内存中的数组，
// inputDevPtr通过调用 在设备上进行处理MyKernel()，并将结果复制outputDevPtr回 的相同部分hostPtr。
// 重叠行为描述了本示例中流如何根据设备的功能重叠。请注意，hostPtr必须指向页面锁定的主机内存才能发生任何重叠。

// 流通过调用来释放`cudaStreanDestory()`
for(int i = 0;i<2;++i)cudaStreanDestory(stream[i]);

// 如果cudaStreamDestroy()调用时设备仍在流中工作，则该函数将立即返回，并且一旦设备完成流中的所有工作，与流关联的资源将自动释放。


