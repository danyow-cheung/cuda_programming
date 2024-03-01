
cudaGetDeviceProperties(&prop,device_id);
size_t size = min(int(prop.l2CacheSize*0.75),prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,size); 

