cudaStreamAttrValue stream_attribute;// Kernel level attributes data structure
node_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(ptr); // Global memory data pointer 
node_attribute.accessPolicyWindow.num_bytes =num_bytes // Number of bytes for persistence access.

node_attribute.accessPolicyWindow,hitRatio = 0.6;// Hint for cache hit ratio
node_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisiting ;// Type of access property on cache hit
node_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;// Type of access property on cache miss.
// set the attributes to a CUDA Graph kernel node of type cudaGraphNode_t
cudaGraphKernelNodeSetAttrinute(node,cudaGraphKernelNodeSetAttributeAccessPolicyWindow,&node_attribute);
