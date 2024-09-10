#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"


__device__ __constant__ float4 TWO_F4 = static_cast<float4>(2.0);
__device__ __constant__ float4 ONE_F4 = static_cast<float4>(1.0);

__device__ void fisheye_srclocs_hip_compute(int i, d_float8 *normX_f8, d_float8 *normY_f8, d_float8 *dist_f8, 
                                            int2 *widthHeight_i2, d_float16 *locSrc_f16)
{
    float dist = dist_f8->f1[i];
    if ((dist >= 0.0) && (dist <= 1.0))
    {
        float newDist = sqrtf(1.0 - dist * dist);
        newDist = (dist + (1.0 - newDist)) * 0.5f;
        if (newDist <= 1.0)
        {
            float theta = atan2f(normY_f8->f1[i], normX_f8->f1[i]);
            float newX = newDist * cosf(theta);
            float newY = newDist * sinf(theta);
            locSrc_f16->f8[0].f1[i] = ((newX + 1) * widthHeight_i2->x) * 0.5f;
            locSrc_f16->f8[1].f1[i] = ((newY + 1) * widthHeight_i2->y) * 0.5f;
        }
    }
}

__device__ void norm_and_dist_hip_compute(int2 *idxy_i2, int2 *widthHeight_i2, d_float8 *normX_f8, d_float8 *normY_f8, d_float8 *dist_f8)
{
    d_float8 increment_f8;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    normY_f8->f4[0] = static_cast<float4>(((static_cast<float>(2 * idxy_i2->y) / widthHeight_i2->y)) - 1);
    normY_f8->f4[1] = normY_f8->f4[0];    
    normX_f8->f4[0] = (TWO_F4 * (static_cast<float4>(idxy_i2->x) + increment_f8.f4[0]) / static_cast<float4>(widthHeight_i2->x)) - ONE_F4;
    normX_f8->f4[1] = (TWO_F4 * (static_cast<float4>(idxy_i2->x) + increment_f8.f4[1]) / static_cast<float4>(widthHeight_i2->x)) - ONE_F4;
    dist_f8->f4[0] = ((normX_f8->f4[0] * normX_f8->f4[0]) + (normY_f8->f4[0] * normY_f8->f4[0]));
    dist_f8->f4[1] = ((normX_f8->f4[1] * normX_f8->f4[1]) + (normY_f8->f4[1] * normY_f8->f4[1]));
    rpp_hip_math_sqrt8(dist_f8, dist_f8);
}

template <typename T>
__global__ void fisheye_pkd_hip_tensor(T *srcPtr,
                                       uint2 srcStridesNH,
                                       T *dstPtr,
                                       uint2 dstStridesNH,
                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int height = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int width = roiTensorPtrSrc[id_z].xywhROI.roiWidth;

    if ((id_y >= height) || (id_x >= width))
        return;

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int2 idxy_i2 = make_int2(id_x, id_y);
    int2 widthHeight_i2 = make_int2(width, height);

    d_float8 normY_f8, normX_f8, dist_f8;
    norm_and_dist_hip_compute(&idxy_i2, &widthHeight_i2, &normX_f8, &normY_f8, &dist_f8);

    d_float16 locSrc_f16;
    locSrc_f16.f8[0].f4[0] = static_cast<float4>(width);
    locSrc_f16.f8[0].f4[1] = locSrc_f16.f8[0].f4[0];
    locSrc_f16.f8[1].f4[0] = static_cast<float4>(height);
    locSrc_f16.f8[1].f4[1] = locSrc_f16.f8[1].f4[0];
    fisheye_srclocs_hip_compute(0, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(1, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(2, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(3, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(4, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(5, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(6, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);
    fisheye_srclocs_hip_compute(7, &normX_f8, &normY_f8, &dist_f8, &widthHeight_i2, &locSrc_f16);

    d_float24 pix_f24;
    rpp_hip_interpolate24_nearest_neighbor_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &pix_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}

template <typename T>
__global__ void fisheye_pln_hip_tensor(T *srcPtr,
                                       uint3 srcStridesNCH,
                                       T *dstPtr,
                                       uint3 dstStridesNCH,
                                       int channelsDst,
                                       RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float8 src1_f8, src2_f8, dst_f8;

    rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src1_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src1_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &src1_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void fisheye_pkd3_pln3_hip_tensor(T *srcPtr,
                                             uint2 srcStridesNH,
                                             T *dstPtr,
                                             uint3 dstStridesNCH,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    d_float24 src1_f24, src2_f24, dst_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src1_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void fisheye_pln3_pkd3_hip_tensor(T *srcPtr,
                                             uint3 srcStridesNCH,
                                             T *dstPtr,
                                             uint2 dstStridesNH,
                                             RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    d_float24 src1_f24, src2_f24, dst_f24;
    rpp_hip_load24_pln3_and_unpack_to_float24_pkd3(srcPtr + srcIdx, srcStridesNCH.y, &src1_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
RppStatus hip_exec_fisheye_tensor(T *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  T *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    Rpp32s globalThreads_x = (dstDescPtr->w + 7) >> 3;
    Rpp32s globalThreads_y = dstDescPtr->h;
    Rpp32s globalThreads_z = dstDescPtr->n;

    if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(fisheye_pkd_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(fisheye_pln_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(fisheye_pkd3_pln3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(fisheye_pln3_pkd3_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               roiTensorPtrSrc);
        }
    }

    return RPP_SUCCESS;
}
