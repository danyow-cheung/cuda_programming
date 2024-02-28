cudaGraph_t graph; // preexisting graph 
Object *object = new Object; // c++ object with possibly notrivial destructor 
cudaUserObject_t cubObject;

cudaUserObjectCreate(
    &cubObject, 
    object,  // use a cuda provided template wrapper for this api ,which supplies a callback to delete the c++ object pointer 
    1, // initiail refcount 
    cudaUserObjectNoDestructorSync // acknowledge that the callback cannot be waited on vida CUDA
);

cudaGraphRetainUserObject(
    graph,
    cubObject,
    1, // number of reference
    cudaGraphUserObjectMove // transfer a reference owned by the caller // don't modify the total reference count
);

