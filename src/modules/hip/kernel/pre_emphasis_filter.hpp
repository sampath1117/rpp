#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ __forceinline__ void pre_emphasis_filter_hip_compute(d_float8 *src1_f8, d_float8 *src2_f8, d_float8 *dst_f8, float4 *coeff_f4)
{
    dst_f8->f4[0] = src1_f8->f4[0] - (*coeff_f4 * src2_f8->f4[0]);
    dst_f8->f4[1] = src1_f8->f4[1] - (*coeff_f4 * src2_f8->f4[1]);
}

__global__ void pre_emphasis_filter_first_eight(float *srcPtr,
                                                uint srcStride,
                                                float *dstPtr,
                                                uint dstStride,
                                                float *coeffTensor,
                                                int *srcLengthTensor,
                                                RpptAudioBorderType borderType)
{
    int id_x = hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= 8)
        return;

    uint srcIdx = (id_z * srcStride) + id_x;
    uint dstIdx = (id_z * dstStride) + id_x;
    float coeff = coeffTensor[id_z];

    d_float8 src1_f8, src2_f8, dst_f8;
    float4 coeff_f4 = static_cast<float4>(coeff);
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src1_f8);

    src2_f8.f1[0] = (borderType == RpptAudioBorderType::CLAMP) ? src1_f8.f1[0] :
                    (borderType == RpptAudioBorderType::REFLECT) ? src1_f8.f1[1] : 0.0f;
    for (int i = 1; i < 8; i++)
        src2_f8.f1[i] = src1_f8.f1[i - 1];

    pre_emphasis_filter_hip_compute(&src1_f8, &src2_f8, &dst_f8, &coeff_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

__global__ void pre_emphasis_filter_tensor(float *srcPtr,
                                                uint srcStride,
                                                float *dstPtr,
                                                uint dstStride,
                                                float *coeffTensor,
                                                int *srcLengthTensor,
                                                RpptAudioBorderType borderType)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x + 1) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x < 8 || id_x >= srcLengthTensor[id_z])
        return;

    uint srcIdx = (id_z * srcStride) + id_x;
    uint dstIdx = (id_z * dstStride) + id_x;
    float coeff = coeffTensor[id_z];

    d_float8 src1_f8, src2_f8, dst_f8;
    float4 coeff_f4 = static_cast<float4>(coeff);
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src1_f8);
    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx - 1, &src2_f8);

    pre_emphasis_filter_hip_compute(&src1_f8, &src2_f8, &dst_f8, &coeff_f4);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
}

RppStatus hip_exec_pre_emphasis_filter_tensor(Rpp32f *srcPtr,
                                              RpptDescPtr srcDescPtr,
                                              Rpp32f *dstPtr,
                                              RpptDescPtr dstDescPtr,
                                              Rpp32f *coeffTensor,
                                              Rpp32s *srcLengthTensor,
                                              RpptAudioBorderType borderType,
                                              rpp::Handle& handle)
{
    Rpp32s globalThreads_x = (dstDescPtr->strides.nStride + 7) >> 3;
    Rpp32s globalThreads_y = 1;
    Rpp32s globalThreads_z = dstDescPtr->n;

    // Launch the kernel for the first eight elements
    hipLaunchKernelGGL(pre_emphasis_filter_first_eight,
                       dim3(1, 1, ceil((float)globalThreads_z / LOCAL_THREADS_Z_1DIM)),
                       dim3(1, 1, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       dstPtr,
                       dstDescPtr->strides.nStride,
                       coeffTensor,
                       srcLengthTensor,
                       borderType);

    hipDeviceSynchronize();

    hipLaunchKernelGGL(pre_emphasis_filter_tensor,
                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((float)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((float)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                       dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       srcDescPtr->strides.nStride,
                       dstPtr,
                       dstDescPtr->strides.nStride,
                       coeffTensor,
                       srcLengthTensor,
                       borderType);

    return RPP_SUCCESS;
}
