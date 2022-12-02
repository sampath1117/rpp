#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - lens_correction device helpers --------------------

__device__ void lens_correction_roi_and_srclocs_hip_compute(int4 *srcRoiPtr_i4, int id_x, int id_y, float zoom, float invCorrectionRadius, d_float16 *locSrc_f16)
{
    int roiHalfWidth = (srcRoiPtr_i4->z - srcRoiPtr_i4->x + 1) >> 1;
    int roiHalfHeight = (srcRoiPtr_i4->w - srcRoiPtr_i4->y + 1) >> 1;

    d_float8 increment_f8, locDst_f8x, locDst_f8y;
    increment_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
    increment_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);
    locDst_f8x.f4[0] = (float4)id_x + increment_f8.f4[0];
    locDst_f8x.f4[1] = (float4)id_x + increment_f8.f4[1];
    locDst_f8y.f4[0] = (float4)id_y;
    locDst_f8y.f4[1] = (float4)id_y;

    locDst_f8x.f4[0] = locDst_f8x.f4[0] - (float4)roiHalfWidth;
    locDst_f8x.f4[1] = locDst_f8x.f4[1] - (float4)roiHalfWidth;
    locDst_f8y.f4[0] = locDst_f8y.f4[0] - (float4)roiHalfHeight;
    locDst_f8y.f4[1] = locDst_f8y.f4[1] - (float4)roiHalfHeight;

    d_float8 distance_f8;
    distance_f8.f1[0] = sqrtf(locDst_f8x.f1[0] * locDst_f8x.f1[0] + locDst_f8y.f1[0] * locDst_f8y.f1[0]) * invCorrectionRadius;
    distance_f8.f1[1] = sqrtf(locDst_f8x.f1[1] * locDst_f8x.f1[1] + locDst_f8y.f1[1] * locDst_f8y.f1[1]) * invCorrectionRadius;
    distance_f8.f1[2] = sqrtf(locDst_f8x.f1[2] * locDst_f8x.f1[2] + locDst_f8y.f1[2] * locDst_f8y.f1[2]) * invCorrectionRadius;
    distance_f8.f1[3] = sqrtf(locDst_f8x.f1[3] * locDst_f8x.f1[3] + locDst_f8y.f1[3] * locDst_f8y.f1[3]) * invCorrectionRadius;
    distance_f8.f1[4] = sqrtf(locDst_f8x.f1[4] * locDst_f8x.f1[4] + locDst_f8y.f1[4] * locDst_f8y.f1[4]) * invCorrectionRadius;
    distance_f8.f1[5] = sqrtf(locDst_f8x.f1[5] * locDst_f8x.f1[5] + locDst_f8y.f1[5] * locDst_f8y.f1[5]) * invCorrectionRadius;
    distance_f8.f1[6] = sqrtf(locDst_f8x.f1[6] * locDst_f8x.f1[6] + locDst_f8y.f1[6] * locDst_f8y.f1[6]) * invCorrectionRadius;
    distance_f8.f1[7] = sqrtf(locDst_f8x.f1[7] * locDst_f8x.f1[7] + locDst_f8y.f1[7] * locDst_f8y.f1[7]) * invCorrectionRadius;

    d_float8 theta_f8;
    theta_f8.f1[0] = (distance_f8.f1[0] == 0) ?  1 : atanf(distance_f8.f1[0]) / distance_f8.f1[0];
    theta_f8.f1[1] = (distance_f8.f1[1] == 0) ?  1 : atanf(distance_f8.f1[1]) / distance_f8.f1[1];
    theta_f8.f1[2] = (distance_f8.f1[2] == 0) ?  1 : atanf(distance_f8.f1[2]) / distance_f8.f1[2];
    theta_f8.f1[3] = (distance_f8.f1[3] == 0) ?  1 : atanf(distance_f8.f1[3]) / distance_f8.f1[3];
    theta_f8.f1[4] = (distance_f8.f1[4] == 0) ?  1 : atanf(distance_f8.f1[4]) / distance_f8.f1[4];
    theta_f8.f1[5] = (distance_f8.f1[5] == 0) ?  1 : atanf(distance_f8.f1[5]) / distance_f8.f1[5];
    theta_f8.f1[6] = (distance_f8.f1[6] == 0) ?  1 : atanf(distance_f8.f1[6]) / distance_f8.f1[6];
    theta_f8.f1[7] = (distance_f8.f1[7] == 0) ?  1 : atanf(distance_f8.f1[7]) / distance_f8.f1[7];

    locSrc_f16->f8[0].f4[0] = (float4)roiHalfWidth;
    locSrc_f16->f8[0].f4[1] = (float4)roiHalfWidth;
    locSrc_f16->f8[1].f4[0] = (float4)roiHalfHeight;
    locSrc_f16->f8[1].f4[1] = (float4)roiHalfHeight;

    locSrc_f16->f1[0] = locSrc_f16->f1[0] + (locDst_f8x.f1[0] * theta_f8.f1[0] * zoom);
    locSrc_f16->f1[1] = locSrc_f16->f1[1] + (locDst_f8x.f1[1] * theta_f8.f1[1] * zoom);
    locSrc_f16->f1[2] = locSrc_f16->f1[2] + (locDst_f8x.f1[2] * theta_f8.f1[2] * zoom);
    locSrc_f16->f1[3] = locSrc_f16->f1[3] + (locDst_f8x.f1[3] * theta_f8.f1[3] * zoom);
    locSrc_f16->f1[4] = locSrc_f16->f1[4] + (locDst_f8x.f1[4] * theta_f8.f1[4] * zoom);
    locSrc_f16->f1[5] = locSrc_f16->f1[5] + (locDst_f8x.f1[5] * theta_f8.f1[5] * zoom);
    locSrc_f16->f1[6] = locSrc_f16->f1[6] + (locDst_f8x.f1[6] * theta_f8.f1[6] * zoom);
    locSrc_f16->f1[7] = locSrc_f16->f1[7] + (locDst_f8x.f1[7] * theta_f8.f1[7] * zoom);

    locSrc_f16->f1[8] = locSrc_f16->f1[8] + (locDst_f8y.f1[0] * theta_f8.f1[0] * zoom);
    locSrc_f16->f1[9] = locSrc_f16->f1[9] + (locDst_f8y.f1[1] * theta_f8.f1[1] * zoom);
    locSrc_f16->f1[10] = locSrc_f16->f1[10] + (locDst_f8y.f1[2] * theta_f8.f1[2] * zoom);
    locSrc_f16->f1[11] = locSrc_f16->f1[11] + (locDst_f8y.f1[3] * theta_f8.f1[3] * zoom);
    locSrc_f16->f1[12] = locSrc_f16->f1[12] + (locDst_f8y.f1[4] * theta_f8.f1[4] * zoom);
    locSrc_f16->f1[13] = locSrc_f16->f1[13] + (locDst_f8y.f1[5] * theta_f8.f1[5] * zoom);
    locSrc_f16->f1[14] = locSrc_f16->f1[14] + (locDst_f8y.f1[6] * theta_f8.f1[6] * zoom);
    locSrc_f16->f1[15] = locSrc_f16->f1[15] + (locDst_f8y.f1[7] * theta_f8.f1[7] * zoom);
}

// -------------------- Set 1 - Bilinear Interpolation --------------------

template <typename T>
__global__ void lens_correction_bilinear_pkd_tensor(T *srcPtr,
                                                    uint2 srcStridesNH,
                                                    T *dstPtr,
                                                    uint2 dstStridesNH,
                                                    float *zoomTensor,
                                                    float *strengthTensor,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = srcRoi_i4.z + 1;
    int height = srcRoi_i4.w + 1;

    if ((id_y >= height) || (id_x >= width))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float zoom = zoomTensor[id_z];
    float strength = strengthTensor[id_z];
    if (strength == 0.0f)
        strength = 0.000001;
    float invCorrectionRadius = strength / sqrtf(width * width + height * height);

    d_float16 locSrc_f16;
    lens_correction_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, zoom, invCorrectionRadius, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

template <typename T>
__global__ void lens_correction_bilinear_pln_tensor(T *srcPtr,
                                                    uint3 srcStridesNCH,
                                                    T *dstPtr,
                                                    uint3 dstStridesNCH,
                                                    int channelsDst,
                                                    float *zoomTensor,
                                                    float *strengthTensor,
                                                    RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = srcRoi_i4.z + 1;
    int height = srcRoi_i4.w + 1;

    if ((id_y >= height) || (id_x >= width))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float zoom = zoomTensor[id_z];
    float strength = strengthTensor[id_z];
    if (strength == 0.0f)
        strength = 0.000001;
    float invCorrectionRadius = strength / sqrtf(width * width + height * height);

    d_float16 locSrc_f16;
    lens_correction_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, zoom, invCorrectionRadius, &locSrc_f16);

    d_float8 dst_f8;
    rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    if (channelsDst == 3)
    {
        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

        srcIdx += srcStridesNCH.y;
        dstIdx += dstStridesNCH.y;

        rpp_hip_interpolate8_bilinear_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    }
}

template <typename T>
__global__ void lens_correction_bilinear_pkd3_pln3_tensor(T *srcPtr,
                                                          uint2 srcStridesNH,
                                                          T *dstPtr,
                                                          uint3 dstStridesNCH,
                                                          float *zoomTensor,
                                                          float *strengthTensor,
                                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = srcRoi_i4.z + 1;
    int height = srcRoi_i4.w + 1;

    if ((id_y >= height) || (id_x >= width))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;

    float zoom = zoomTensor[id_z];
    float strength = strengthTensor[id_z];
    if (strength == 0.0f)
        strength = 0.000001;
    float invCorrectionRadius = strength / sqrtf(width * width + height * height);

    d_float16 locSrc_f16;
    lens_correction_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, zoom, invCorrectionRadius, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pkd3(srcPtr + srcIdx, srcStridesNH.y, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pkd3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
}

template <typename T>
__global__ void lens_correction_bilinear_pln3_pkd3_tensor(T *srcPtr,
                                                          uint3 srcStridesNCH,
                                                          T *dstPtr,
                                                          uint2 dstStridesNH,
                                                          float *zoomTensor,
                                                          float *strengthTensor,
                                                          RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];
    int width = srcRoi_i4.z + 1;
    int height = srcRoi_i4.w + 1;

    if ((id_y >= height) || (id_x >= width))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x * 3;

    float zoom = zoomTensor[id_z];
    float strength = strengthTensor[id_z];
    if (strength == 0.0f)
        strength = 0.000001;
    float invCorrectionRadius = strength / sqrtf(width * width + height * height);

    d_float16 locSrc_f16;
    lens_correction_roi_and_srclocs_hip_compute(&srcRoi_i4, id_x, id_y, zoom, invCorrectionRadius, &locSrc_f16);

    d_float24 dst_f24;
    rpp_hip_interpolate24_bilinear_pln3(srcPtr + srcIdx, &srcStridesNCH, &locSrc_f16, &srcRoi_i4, &dst_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &dst_f24);
}

// -------------------- Set 2 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_lens_correction_tensor(T *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          T *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          RpptInterpolationType interpolationType,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::XYWH)
        hip_exec_roi_converison_xywh_to_ltrb(roiTensorPtrSrc, handle);

    int localThreads_x = 16;
    int localThreads_y = 16;
    int localThreads_z = 1;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    if (interpolationType == RpptInterpolationType::BILINEAR)
    {
        if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            hipLaunchKernelGGL(lens_correction_bilinear_pkd_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            hipLaunchKernelGGL(lens_correction_bilinear_pln_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr,
                               make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                               dstPtr,
                               make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                               dstDescPtr->c,
                               handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                               handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                               roiTensorPtrSrc);
        }
        else if ((srcDescPtr->c == 3) && (dstDescPtr->c == 3))
        {
            if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
            {
                hipLaunchKernelGGL(lens_correction_bilinear_pkd3_pln3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                   handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                   roiTensorPtrSrc);
            }
            else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
            {
                globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
                hipLaunchKernelGGL(lens_correction_bilinear_pln3_pkd3_tensor,
                                   dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                   dim3(localThreads_x, localThreads_y, localThreads_z),
                                   0,
                                   handle.GetStream(),
                                   srcPtr,
                                   make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                                   dstPtr,
                                   make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                                   handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                                   handle.GetInitHandle()->mem.mgpu.floatArr[1].floatmem,
                                   roiTensorPtrSrc);
            }
        }
    }

    return RPP_SUCCESS;
}
