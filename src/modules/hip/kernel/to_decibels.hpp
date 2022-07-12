#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void to_decibels_hip_compute(d_float8 *src_f8, d_float8 *dst_f8, float minRatio, float multiplier, float invReferenceMagnitude)
{
    dst_f8->f1[0] = multiplier * log10(max(minRatio, src_f8->f1[0] * invReferenceMagnitude));
    dst_f8->f1[1] = multiplier * log10(max(minRatio, src_f8->f1[1] * invReferenceMagnitude));
    dst_f8->f1[2] = multiplier * log10(max(minRatio, src_f8->f1[2] * invReferenceMagnitude));
    dst_f8->f1[3] = multiplier * log10(max(minRatio, src_f8->f1[3] * invReferenceMagnitude));
    dst_f8->f1[4] = multiplier * log10(max(minRatio, src_f8->f1[4] * invReferenceMagnitude));
    dst_f8->f1[5] = multiplier * log10(max(minRatio, src_f8->f1[5] * invReferenceMagnitude));
    dst_f8->f1[6] = multiplier * log10(max(minRatio, src_f8->f1[6] * invReferenceMagnitude));
    dst_f8->f1[7] = multiplier * log10(max(minRatio, src_f8->f1[7] * invReferenceMagnitude));
}

__global__ void to_decibels_tensor(float *srcPtr,
                                   uint2 srcStridesNH,
                                   float *dstPtr,
                                   long long *samplesPerChannelTensor,
                                   float cutOffDB,
                                   float multiplier,
                                   float referenceMagnitude)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= samplesPerChannelTensor[id_z])
    {
        return;
    }

    uint loc = (id_z * srcStridesNH.x) + id_x;
    bool referenceMax = (referenceMagnitude == 0.0) ? false : true;
    if(!referenceMax)
    {
        referenceMagnitude = 1.0;
    }
    float invreferenceMagnitude = (1.0f / referenceMagnitude);
    float minRatio = pow(10, cutOffDB / multiplier);

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + loc, &src_f8);
    to_decibels_hip_compute(&src_f8, &dst_f8, minRatio, multiplier, invreferenceMagnitude);
    rpp_hip_pack_float8_and_store8(dstPtr + loc, &dst_f8);
}

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      Rpp64s *samplesPerChannelTensor,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = 1;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = srcDescPtr->w;
    int globalThreads_y = srcDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    hipLaunchKernelGGL(to_decibels_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       samplesPerChannelTensor,
                       cutOffDB,
                       multiplier,
                       referenceMagnitude);

    return RPP_SUCCESS;
}
