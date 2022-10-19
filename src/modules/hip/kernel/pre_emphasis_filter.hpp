#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__global__ void pre_emphasis_filter_tensor(float *srcPtr,
                                           uint2 srcStridesNH,
                                           float *dstPtr,
                                           uint2 dstStridesNH,
                                           int *srcSizeTensor,
                                           float *coeffTensor,
                                           RpptAudioBorderType borderType)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcSizeTensor[id_z])
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + id_x;
    float coeff = coeffTensor[id_z];
    if(id_x == 0)
    {
        if(borderType == RpptAudioBorderType::ZERO)
            dstPtr[dstIdx] = srcPtr[srcIdx];
        else
        {
            float border = (borderType == RpptAudioBorderType::CLAMP) ? srcPtr[srcIdx] : srcPtr[srcIdx + 1];
            dstPtr[dstIdx] = srcPtr[srcIdx] - coeff * border;
        }
    }
    else
        dstPtr[dstIdx] = srcPtr[srcIdx] - coeff * srcPtr[srcIdx - 1];
}

RppStatus hip_exec_pre_emphasis_filter_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              RpptAudioBorderType borderType,
                                              rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Z;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = srcDescPtr->strides.hStride;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(pre_emphasis_filter_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       borderType);

    return RPP_SUCCESS;
}
