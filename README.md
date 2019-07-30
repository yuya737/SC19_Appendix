# SC19_Archive

OpenMP
```console
g++ -O3 -std=c++11 -mcpu=power9 -mtune=power9 -o clippingOMP  clippingCPU.cpp -fopenmp  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -DTIMING -lgomp
```

TBB
```console
g++ -O3 -std=c++11  -o clippingTBB  clippingCPU.cpp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB  -ltbb
```

Single GPU - CudaDefaultAllocation
```console
nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_CudaDefaultAllocation  clippingGPU_rmm_CudaDefaultAllocation.cu -lrmm -I<path-to-rmm-headers>  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70
```

Single GPU - PoolAllocation
```console
nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_PoolAllocation  clippingGPU_rmm_PoolAllocation.cu -lrmm -I<path-to-rmm-headers>  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70
```

Single GPU - CudaManaged
```console
nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_cudaManaged  clippingGPU_rmm_cudaManaged.cu -lrmm -I<path-to-rmm-headers> -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70
```

Single GPU - CudaManagedPool
```console
nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_cudaManagedPool  clippingGPU_rmm_cudaManagedPool.cu -lrmm -I<path-to-rmm-headers>  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70
```

Multi GPU - CudaDefaultAllocation
```console
nvcc --default-stream per-thread  -O3 -std=c++11  -o clipping_multiGPU_rmm_CudaDefaultAllocation  clipping_multiGPU_rmm_CudaDefaultAllocation.cu -lrmm -I<path-to-rmm-headers> -gencode arch=compute_70,code=sm_70 
```

Multi GPU - CudaManaged
```console
nvcc --default-stream per-thread  -O3 -std=c++11  -o clipping_multiGPU_rmm_cudaManaged  clipping_multiGPU_rmm_cudaManaged.cu -lrmm -I<path-to-rmm-headers> -gencode arch=compute_70,code=sm_70
```



