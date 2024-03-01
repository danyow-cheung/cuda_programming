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


cudaGraphRetainUserObject(
    graph,cubObject,1,cudaGraphUserObjectMove //Transfer a reference owned by the caller (do
                             // not modify the total reference count)
);

//no more reference owned by this thread,no need to call release api
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec,graph,nullptr,nullptr,0);

cudaGraphDestroy(graph);
cudaGraphLaunch(graphExec,0);
cudaGraphExecDestory(graphExec);

cudaStreamSynchronize(0);
