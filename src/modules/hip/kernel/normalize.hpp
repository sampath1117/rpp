#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void normalize_audio_tensor(float *srcPtr,
                                       uint2 srcStridesNH,
                                       float *dstPtr,
                                       uint2 dstStridesNH,
                                       float *meanTensor,
                                       float *stdDevTensor)
{
    // int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    // int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    // int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    // if (id_x >= srcSizeTensor[id_z])
    // {
    //     return;
    // }

    // uint srcIdx = (id_z * srcStridesNH.x) + id_x;
    // uint dstIdx = (id_z * dstStridesNH.x) + id_x;
}

__global__ void compute_mean(float *srcPtr,
                             uint2 srcStridesNH,
                             float *meanTensor,
                             int *strides,
                             int *dims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int batchLoc = id_z * 2;
    int dim0 = dims[batchLoc];
    int dim1 = dims[batchLoc + 1];
    int stride0 = strides[batchLoc];
    int stride1 = strides[batchLoc + 1];

    float meanVal = 0;
    if(id_x < dim0)
    {
        uint srcIdx = (id_z * srcStridesNH.x);
        for(int i = 0; i < dim1; i++)
        {
           meanVal += srcPtr[srcIdx + i * stride0 + id_x];
        }
        meanVal /= dim1;
        meanTensor[id_z * stride0 + id_x] = meanVal;
    }
}

__global__ void compute_std(float *srcPtr,
                            float *stdDevTensor,
                            int *strides,
                            int *dims)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

}

RppStatus hip_exec_normalize_audio_tensor(Rpp32f *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          Rpp32f *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          Rpp32s axisMask,
                                          Rpp32f scale,
                                          Rpp32f shift,
                                          Rpp32f epsilon,
                                          Rpp32s ddof,
                                          rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Z;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = srcDescPtr->strides.hStride;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    // if(!mean)
    // {
        int block_x = LOCAL_THREADS_X;
        int block_y = 1;
        int block_z = LOCAL_THREADS_Z;

        int grid_x;
        if(axisMask == 1)
            grid_x = srcDescPtr->w;
        else
            grid_x = srcDescPtr->h;
        int grid_y = 1;
        int grid_z = handle.GetBatchSize();
        hipLaunchKernelGGL(compute_mean,
                           dim3(ceil((float)grid_x/block_x), ceil((float)grid_y/block_y), ceil((float)grid_z/block_z)),
                           dim3(block_x, block_y, block_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           handle.GetInitHandle()->mem.mgpu.meanArr.floatmem,
                           handle.GetInitHandle()->mem.mgpu.intArr[1].intmem,  // sride values
                           handle.GetInitHandle()->mem.mgpu.intArr[2].intmem); // Reduction Dimensions

        hipDeviceSynchronize();
        float *temp_mean = (float *)malloc(grid_x * sizeof(Rpp32f) *  srcDescPtr->n);
        hipMemcpy((void *)temp_mean, handle.GetInitHandle()->mem.mgpu.meanArr.floatmem, grid_x * sizeof(Rpp32f) *  srcDescPtr->n, hipMemcpyDeviceToHost);
        printf("mean values are \n");
        for(int i = 0; i < grid_x; i++)
        {
            printf("%f\n", temp_mean[i]);
        }
    // }

    // hipLaunchKernelGGL(normalize_audio_tensor,
    //                    dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
    //                    dim3(localThreads_x, localThreads_y, localThreads_z),
    //                    0,
    //                    handle.GetStream(),
    //                    srcPtr,
    //                    make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
    //                    dstPtr,
    //                    make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
    //                    handle.GetInitHandle()->mem.mgpu.meanArr.floatmem,
    //                    handle.GetInitHandle()->mem.mgpu.meanArr.floatmem,
    //                    borderType);

    return RPP_SUCCESS;
}
