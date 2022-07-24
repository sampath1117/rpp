#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void pre_emphasis_filter_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *coeff_f4)
{
    dst_f8->f4[0] = src1_f8->f4[0] - (*coeff_f4) * src2_f8->f4[0];
    dst_f8->f4[1] = src1_f8->f4[1] - (*coeff_f4) * src2_f8->f4[1];
}

__global__ void pre_emphasis_filter_tensor(float *srcPtr,
                                           uint2 srcStridesNH,
                                           float *dstPtr,
                                           int *srcSizeTensor,
                                           float *coeffTensor,
                                           uint borderType)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcSizeTensor[id_z])
    {
        return;
    }

    uint loc = (id_z * srcStridesNH.x) + id_x;
    float coeff = coeffTensor[id_z];
    if(id_x == 0)
    {
        if(borderType == 0)
            dstPtr[loc] = srcPtr[loc];
        else
        {
            float border = (borderType == 1) ? srcPtr[1] : srcPtr[0];
            dstPtr[loc] = srcPtr[loc] - coeff * border;
        }
    }
    else
    {
        d_float8 src1_f8, src2_f8, dst_f8;
        rpp_hip_load8_and_unpack_to_float8(srcPtr + loc, &src1_f8);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + loc - 1, &src2_f8);
        float4 coeff_f4 = (float4)coeff;
        pre_emphasis_filter_hip_compute(&src1_f8, &src2_f8, &dst_f8, &coeff_f4);
        rpp_hip_pack_float8_and_store8(dstPtr + loc, &dst_f8);
    }
}

RppStatus hip_exec_pre_emphasis_filter_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptAudioBorderType borderType,
                                              rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
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
                       handle.GetInitHandle()->mem.mgpu.intArr[0].intmem,
                       handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                       borderType);

    return RPP_SUCCESS;
}
