# CUDA_programming

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

如何编译运行代码
https://support.cs.wm.edu/index.php/tips-and-tricks/hello-world-program-with-cuda
如果是在windows下运行则报错
`nvcc .\tail_launch.cu -o tail_launch
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH`
> 解决办法:https://stackoverflow.com/questions/8125826/error-compiling-cuda-from-command-prompt

本机路径
`E:\VisualStudio_2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe`
会话添加环境变量
`$env:Path += ";E:\VisualStudio_2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"`


第三章basically学完了，如果有新增的知识添加到new文件夹
