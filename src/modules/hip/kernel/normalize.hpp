#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void compute_mean_and_std(float *srcPtr,
                                     uint3 srcStrides,
                                     float *meanTensor,
                                     float *stdDevTensor,
                                     int *reductionDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int paramLoc = id_z * 2;
    int dim0 = reductionDims[paramLoc];
    int dim1 = reductionDims[paramLoc + 1];

    float meanVal = 0.0f;
    float stdVal = 0.0f;
    if(id_x < dim0)
    {
        uint tempIdx = (id_z * srcStrides.x) + id_x * srcStrides.z;
        uint dstIdx = id_z * srcStrides.y + id_x;
        float sumOfElements = 0.0f;
        float sumOfSquaredElements = 0.0f;
        for(int i = 0; i < dim1; i++)
        {
           uint srcIdx = tempIdx + i * srcStrides.y;
           float val = srcPtr[srcIdx];
           sumOfElements += val;
           sumOfSquaredElements += (val * val);
        }
        meanVal = sumOfElements / dim1;
        meanTensor[dstIdx] = meanVal;

        stdVal = (sumOfSquaredElements + (dim1 * meanVal * meanVal) - (2 * meanVal * sumOfElements)) / dim1;
        stdVal = (!stdVal) ? 0.0f : 1.0f / sqrt(stdVal);
        stdDevTensor[dstIdx] = stdVal;
    }
}

__global__ void compute_mean(float *srcPtr,
                             uint3 srcStrides,
                             float *meanTensor,
                             int *reductionDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int paramLoc = id_z * 2;
    int dim0 = reductionDims[paramLoc];
    int dim1 = reductionDims[paramLoc + 1];

    float meanVal = 0;
    if(id_x < dim0)
    {
        uint tempIdx = (id_z * srcStrides.x) + id_x * srcStrides.z;
        uint dstIdx = id_z * srcStrides.y + id_x;
        for(int i = 0; i < dim1; i++)
        {
           uint loc = tempIdx + i * srcStrides.y;
           meanVal += srcPtr[loc];
        }
        meanVal /= dim1;
        meanTensor[dstIdx] = meanVal;
    }
}

__global__ void compute_std(float *srcPtr,
                            uint3 srcStrides,
                            float *meanTensor,
                            float *stdDevTensor,
                            int *reductionDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int paramLoc = id_z * 2;
    int dim0 = reductionDims[paramLoc];
    int dim1 = reductionDims[paramLoc + 1];

    float stdVal = 0;
    if(id_x < dim0)
    {
        uint tempIdx = (id_z * srcStrides.x) + id_x * srcStrides.z;
        uint dstIdx = id_z * srcStrides.y + id_x;
        for(int i = 0; i < dim1; i++)
        {
           uint loc = tempIdx + i * srcStrides.y;
           float diff = srcPtr[loc] - meanTensor[dstIdx];
           stdVal += (diff * diff);
        }
        stdVal /= dim1;
        stdVal = (!stdVal) ? 0.0f : 1.0f / sqrt(stdVal);
        stdDevTensor[dstIdx] = stdVal;
    }
}

__global__ void normalize_audio_tensor(float *srcPtr,
                                       uint3 srcStrides,
                                       float *dstPtr,
                                       float *meanTensor,
                                       float *stdDevTensor,
                                       int *reductionDims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int paramLoc = id_z * 2;
    int dim0 = reductionDims[paramLoc];
    int dim1 = reductionDims[paramLoc + 1];
    float mean = meanTensor[id_z * srcStrides.y + id_x];
    float invStdDev = stdDevTensor[id_z * srcStrides.y + id_x];

    if(id_x < dim0)
    {
        uint tempIdx = (id_z * srcStrides.x) + id_x * srcStrides.z;
        for(int i = 0; i < dim1; i++)
        {
            uint srcIdx = tempIdx + i * srcStrides.y;
            dstPtr[srcIdx] = (srcPtr[srcIdx] - mean) * invStdDev;
        }
    }
}

RppStatus hip_exec_normalize_audio_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s axisMask,
                                          Rpp32f mean,
                                          Rpp32f stdDev,
                                          Rpp32f scale,
                                          Rpp32f shift,
                                          Rpp32f epsilon,
                                          Rpp32s ddof,
                                          rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;

    int globalThreads_x;
    uint stride1, stride2;
    if(axisMask == 1)
    {
        globalThreads_x = srcDescPtr->w;
        stride1 = srcDescPtr->strides.hStride;
        stride2 = srcDescPtr->strides.wStride;
    }
    else if(axisMask == 2)
    {
        globalThreads_x = srcDescPtr->h;
        stride1 = srcDescPtr->strides.wStride;
        stride2 = srcDescPtr->strides.hStride;
    }
    else if(axisMask == 3)
    {
        globalThreads_x = 1;
        stride1 = srcDescPtr->strides.wStride;
        stride2 = srcDescPtr->strides.wStride;
    }
    int globalThreads_y = 1;
    int globalThreads_z = handle.GetBatchSize();

    Rpp32f *d_mean, *d_stdDev;

    hipMalloc(&d_mean, globalThreads_z * globalThreads_x * sizeof(Rpp32f));
    hipMalloc(&d_stdDev, globalThreads_z * globalThreads_x * sizeof(Rpp32f));

    if((!mean) && (!stdDev))
    {
        hipLaunchKernelGGL(compute_mean_and_std,
                           dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y), ceil((float)globalThreads_z / localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, stride1, stride2),
                           d_mean,
                           d_stdDev,
                           handle.GetInitHandle()->mem.mgpu.intArr[0].intmem); // Reduction Dimensions
        hipDeviceSynchronize();
    }
    else if(!(mean) && (stdDev))
    {
        hipLaunchKernelGGL(compute_mean,
                           dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y), ceil((float)globalThreads_z / localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, stride1, stride2),
                           d_mean,
                           handle.GetInitHandle()->mem.mgpu.intArr[0].intmem); // Reduction Dimensions
        hipDeviceSynchronize();
    }
    else if((mean) && (!stdDev))
    {
        hipLaunchKernelGGL(compute_std,
                           dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y), ceil((float)globalThreads_z / localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, stride1, stride2),
                           d_mean,
                           d_stdDev,
                           handle.GetInitHandle()->mem.mgpu.intArr[0].intmem); // Reduction Dimensions
        hipDeviceSynchronize();
    }

    hipLaunchKernelGGL(normalize_audio_tensor,
                       dim3(ceil((float)globalThreads_x / localThreads_x), ceil((float)globalThreads_y / localThreads_y), ceil((float)globalThreads_z / localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint3(srcDescPtr->strides.nStride, stride1, stride2),
                       dstPtr,
                       d_mean,
                       d_stdDev,
                       handle.GetInitHandle()->mem.mgpu.intArr[0].intmem); // Reduction Dimensions

    hipFree(d_mean);
    hipFree(d_stdDev);

    return RPP_SUCCESS;
}