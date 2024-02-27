__global__ void primary_kernel(){
    //initial work that should finish before starting secondart kernel
    // trigger the seconardy kernel
    cudaTriggerProgrammaticLaunchCompletion();
    // work that can coincide with the secondary kernel
}

__global__ void secondary_kernel(){
    // independent work 
    // will  block until all primary kernels the secondary kernel is dependent on have completed and flushed results to global memory
    cudaGridDependencySynchronize();
    // dependent work
}

cudaLaunchAttribute attribute[1];
attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
attribute[0].val.programmaticsStreamSerializationAllowed = 1;
configSecondary.attrs = attribute;
configSecondary.numAttrs = 1;
primary_kernel<<<grid_dim,block_dim,0,stream>>>();
cudaLaunchKernelEx(&configSecondary,secondary_kernel);
