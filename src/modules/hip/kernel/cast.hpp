#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"


template <typename T, typename U>
__device__ __forceinline__ U cast_bit_depth(T *src,U *dst, uint idx)
{
    return static_cast<U>(src[idx]);
}

template <typename T, typename U>
__global__ void cast_hip_tensor(T *srcPtr,
                                uint2 srcStridesNH,
                                U *dstPtr,
                                uint2 dstStridesNH,
                                uint2 maxDim)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= maxDim.y) || (id_x >= maxDim.x * 3))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + (id_x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;

    for(int i=0; i<8; i++)
    {
        dstPtr[srcIdx + i] = cast_bit_depth(srcPtr, dstPtr, srcIdx + i);
    }
}

template <typename T, typename U>
RppStatus hip_exec_cast_tensor(T *srcPtr,
                               RpptDescPtr srcDescPtr,
                               U *dstPtr,
                               RpptDescPtr dstDescPtr,
                               rpp::Handle& handle)
{
    int globalThreads_x = ((dstDescPtr->strides.hStride * dstDescPtr->c) + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    hipLaunchKernelGGL(cast_hip_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                       dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       make_uint2(srcDescPtr->w, srcDescPtr->h));

    return RPP_SUCCESS;
}
