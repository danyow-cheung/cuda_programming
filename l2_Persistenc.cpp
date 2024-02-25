cudaStream_t stream;
cudaStreamCreate(&stream); // 创建cuda流式

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop,device_id); // cuda设备属性向量
size_t size = min(int(prop.l2CacheSize*0.75),prop.persistingL2CacheMaxSize); // 输入GPu属性
cudaDeviceSetLimit (cudaLimitPersistingL2CacheSize,size);  ///留出3/4的二級緩存用於持久訪問或允許的最大值


size_t window_size = min(prop.accessPolicyMaxWindowSize,num_bytes);//選擇用戶定義的最小num_bytes和最大視窗大小。

cudaStreamAttrValue  stream_attribute; // 流級内容資料結構
stream_attribute.accessPolicyWindow.base_prt = reinterpret_cast<void*>(data1); // 全局显存内容指针
stream_attribute.accessPolicyWindow.num_bytes = window_size; // 持久性訪問的位元組數
stream_attribute.accessPolicyWindow.hitRatio = 0.6;//緩存命中率提示
stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; //持久属性
stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; //緩存未命中時的訪問内容類型


cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&stream_attribute); //Set the attributes to a CUDA Stream

for (int i = 0;i<10;i++){
    cuda_kernelA<<<grid_size,block_size,0,stream>>>(data1);
}
//內核多次使用此數據1
//[data1+num_bytes）受益於L2持久性
//同一流中的不同內核也可以受益於
//數據1的持久性

cuda_kernelB<<<grid_size,block_size,0,stream>>>(data1);// Setting the window size to 0 disable it

stream_attribute.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream,cudaStreamAttributeAccessPolicyWindow,&stream_attribute);//Overwrite the access policy attribute to a CUDA Stream
cudaCtxResetPersistingL2Cache();//Remove any persistent lines in L2

cuda_kernel<<<grid_size,block_size,0,stream>>>(data2);// data2 can now benefit from full L2 in normal mode
