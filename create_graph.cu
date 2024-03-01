//3,2,8,7,3 Creating a Graph Using Stream Capture
cudaGraph_t graph;
cudaStreamBeginCapture(stream);

kernel_A<<< ..., stream >>>(...);
kernel_B<<< ..., stream >>>(...);
libraryCall(stream);
kernel_C<<< ..., stream >>>(...);

cudaStreamEndCapture(stream, &graph);

// create the graph -it starts out empty 
cudaGraphCreate(&graph,0);


// For the purpose of this example, we'll create
// the nodes separately from the dependencies to
// demonstrate that it can be done in two stages.
// Note that dependencies can also be specified
// at node creation.
cudaGraphAddKernelNode(&a,graph,NULL,0,&nodeParams);
cudaGraphAddKernelNode(&b,graph,NULL,0,&nodeParams);
cudaGraphAddKernelNode(&c,graph,NULL,0,&nodeParams);
cudaGraphAddKernelNode(&d,graph,NULL,0,&nodeParams);

// now set up dependenvvies on each node 
cudaGraphAddDependencies(graph, &a, &b, 1);     // A->B
cudaGraphAddDependencies(graph, &a, &c, 1);     // A->C
cudaGraphAddDependencies(graph, &b, &d, 1);     // B->D
cudaGraphAddDependencies(graph, &c, &d, 1);     // c->D

