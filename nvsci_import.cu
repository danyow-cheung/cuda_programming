NvSciBufObj createNvSciBufObject(){
    //raw buffer attributes for cuda 
    NvSciBufType bufType = NvSciBufType_RawBuffer;
    uint64_t rawsize = SIZE;
    uint64_t align = 0 ;
    bool cpuaccess_flag = true;
    NvSciBufAttrValAccessPerm perm= NvSciBufAccessPerm_ReadWrite;

    
}