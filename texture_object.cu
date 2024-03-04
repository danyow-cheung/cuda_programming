strcut cudaTextureDesc{
    enum cudaTextureAddressMode addresMode[3];
    enum cudaTextureFilterMode filterMode;
    enum cudaTextureReadMoode readMode;
    int                       sRGB;
    int                       normalizedCoords;
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
};

