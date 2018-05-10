# CudaBilateralFilterImpl
This project aims to compare and optimize the performance of different implementation of bilateral filter.
There is a offical demo from cuda release (located at samples/3_Imaging/bilateralFilter), but this demo only works on 8-bit & 4-channels images
and seriosly it is a little slow. This project try to use skills like lookup table and share memory to improve the performance.

#Developing Enviroments (Optional)
Ubuntu 16.04
Nsight
Navidia Quadro K620

# Requirements
CUDA
OpenCV 2.4.13+

# Implementation description
several implementations are compared:
1. OpenCV's CPU implementation (only support 8u and 32f type of data)
2. Cuda implementation, use global memory to access pixel values (compare the 8u, 16u, 32f type of data)
3. Cuda implementation, use only texture memory to access pixel values 
4. Cuda implementation, use only texture memory to access pixel values, use lookup table to get gaussian weights
5. Cuda implementation, use both texture memory and share memory to access pixel values, use lookup table to get gaussian weights

#Compare Result
nvprof shows:
![image](https://github.com/tstc435/CudaBilateralFilterImpl/raw/master/images/nvprof_of_bilateral_filter.png)

time consumption statistic:
![image](https://github.com/tstc435/CudaBilateralFilterImpl/raw/master/images/compare_result.JPG)

#References
[1] https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
[2] https://github.com/aashikgowda/Bilateral-Filter-CUDA