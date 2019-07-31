# SC19_Appendix

## Prerequisites
- RAPIDS memory manager https://github.com/rapidsai/rmm
- NVIDIA Thrust https://github.com/thrust/thrust
- Intel TBB (Thread Building Blocks) https://www.threadingbuildingblocks.org/
- GNU C/C++ compiler v6.x+
- CUDA 10.0+

NVIDIA Driver Version: 418.67

All code was run on SUMMIT supercomputer at Oak Ridge National Laboratory, TN

## Compilation instructions

- OpenMP
  - `g++ -O3 -std=c++11 -mcpu=power9 -mtune=power9 -o clippingOMP  clippingCPU.cpp -fopenmp  -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP -DTIMING -lgomp`

- TBB
  - `g++ -O3 -std=c++11  -o clippingTBB  clippingCPU.cpp -DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_TBB  -ltbb`

- Single GPU - CudaDefaultAllocation
  - `nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_CudaDefaultAllocation  clippingGPU_rmm_CudaDefaultAllocation.cu -lrmm   -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70`


- Single GPU - PoolAllocation
  - `nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_PoolAllocation  clippingGPU_rmm_PoolAllocation.cu -lrmm   -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70`

- Single GPU - CudaManaged
  - `nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_cudaManaged  clippingGPU_rmm_cudaManaged.cu -lrmm  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70`

- Single GPU - CudaManagedPool
  - `nvcc -O3 -std=c++11 -v -o clippingGPU_rmm_cudaManagedPool  clippingGPU_rmm_cudaManagedPool.cu -lrmm   -gencode arch=compute_52,code=sm_52 -gencode arch=compute_70,code=sm_70`

- Multi GPU - CudaDefaultAllocation
  - `nvcc --default-stream per-thread  -O3 -std=c++11  -o clipping_multiGPU_rmm_CudaDefaultAllocation  clipping_multiGPU_rmm_CudaDefaultAllocation.cu -lrmm  -gencode arch=compute_70,code=sm_70 `

- Multi GPU - CudaManaged
  - `nvcc --default-stream per-thread  -O3 -std=c++11  -o clipping_multiGPU_rmm_cudaManaged  clipping_multiGPU_rmm_cudaManaged.cu -lrmm  -gencode arch=compute_70,code=sm_70`
  
## Execution instructions

To run the OpenMP version, for example,
`clippingOMP 400 400 10000 30` would specify a transversial clipping of a dataset of, x_size=400, y_size=400, z_size=10000 for 30 iterations.







