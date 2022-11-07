#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void to_decibels_tensor(float *srcPtr,
                                   uint2 srcStridesNH,
                                   float *dstPtr,
                                   uint2 dstStridesNH,
                                   RpptImagePatchPtr srcDims,
                                   float minRatio,
                                   float multiplier,
                                   float referenceMagnitude,
                                   float *maxValues)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_x >= srcDims[id_z].width) || (id_y >= srcDims[id_z].height))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    referenceMagnitude = (referenceMagnitude == 0.0) ? maxValues[id_z] : referenceMagnitude;
    float invReferenceMagnitude = (1.0f / referenceMagnitude);
    dstPtr[dstIdx] = multiplier * log10f(fmaxf(minRatio, srcPtr[srcIdx] * invReferenceMagnitude));
}


__global__ void get_max(float *srcPtr,
                        RpptImagePatchPtr srcDims,
                        uint2 srcStridesNH,
                        float *maxValues,
                        float defaultMin,
                        int *mutex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    int threadIdx = hipThreadIdx_x;
    int threadIdy = hipThreadIdx_y;
    int posInBlock = threadIdy * hipBlockDim_x + threadIdx;

    // Store block data in shared memory
    extern __shared__ float shm[];
    shm[posInBlock] = ((id_x >= srcDims[id_z].width) || (id_y >= srcDims[id_z].height))? defaultMin: srcPtr[srcIdx];
    __syncthreads();

    // Do reduction
    for(int s = (hipBlockDim_x * hipBlockDim_y) / 2; s > 0; s >>= 1)
    {
       if(posInBlock < s)
            shm[posInBlock] = fmaxf(shm[posInBlock], shm[posInBlock + s]);
        __syncthreads();
    }

    // Get global maximum across blocks
    if(threadIdx == 0 && threadIdy == 0)
    {
		while(atomicCAS(mutex, 0, 1) != 0);  // lock
		maxValues[id_z] = fmaxf(maxValues[id_z], shm[0]);
		atomicExch(mutex, 0);  // unlock
	}
}

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr srcDims,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle)
{
    int localThreads_x;
    if(dstDescPtr->w == 1)
        localThreads_x = 1; // For 1D input set number of threads for x direction in a block as 1
    else
        localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if(referenceMagnitude == 0.0)
    {
        int *mutex = nullptr;
        hipMalloc((void **)&mutex, sizeof(int));
        hipMemset(mutex, 0, sizeof(int));
        hipLaunchKernelGGL(get_max,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           localThreads_x * localThreads_y * sizeof(float),
                           handle.GetStream(),
                           srcPtr,
                           srcDims,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           -std::numeric_limits<float>::max(),
                           mutex);
        hipDeviceSynchronize();
        hipFree(mutex);
    }

    float minRatio = powf(10, cutOffDB / multiplier);
    hipLaunchKernelGGL(to_decibels_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       srcDims,
                       minRatio,
                       multiplier,
                       referenceMagnitude,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);

    return RPP_SUCCESS;
}
