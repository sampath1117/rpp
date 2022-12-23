#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void glitch_pln_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint3 dstStridesNCH,
                                      int channelDst,
                                      unsigned int *x_offset_r,
                                      unsigned int *y_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *y_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_b,
                                      RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    
    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }
    int x_r, y_r, x_g, y_g, x_b, y_b;

    x_r = id_x + x_offset_r[id_z];
    y_r = id_y + y_offset_r[id_z];

    x_g = id_x + x_offset_g[id_z];
    y_g = id_y + y_offset_g[id_z];

    x_b = id_x + x_offset_b[id_z];
    y_b = id_y + y_offset_b[id_z];

    int srcIdx, dstIdx;
    int srcIdx_r, srcIdx_g, srcIdx_b, dstIdx_r, dstIdx_g, dstIdx_b;
    d_float8 srcR_f8, srcG_f8, srcB_f8;
    d_float24 pix_f24;

    srcIdx = (id_z * srcStridesNCH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    dstIdx_r = dstIdx;
    dstIdx_g = dstIdx_r + srcStridesNCH.y;
    dstIdx_b = dstIdx_g + srcStridesNCH.y;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &pix_f24);

    if((y_r >= 0) && (y_r <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        srcIdx_r = (id_z * srcStridesNCH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        if(id_x == 0 && id_y == 0)
            printf(" \n %d %d %d", srcIdx_r, x_r, y_r);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_r, &srcR_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &srcR_f8);
    }
    
    if((y_g >= 0) && (y_g <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        srcIdx_g = (id_z * srcStridesNCH.x) + srcStridesNCH.y + ((y_g + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_g + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        if(id_x == 0 && id_y == 0)
            printf(" \n %d %d %d", srcIdx_g, x_g, y_g);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_g, &srcG_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_g , &srcG_f8);
    }

    if((y_b >= 0) && (y_b <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        srcIdx_b = (id_z * srcStridesNCH.x) + (srcStridesNCH.y * 2) + ((y_b + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_b + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_b, &srcB_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_b, &srcB_f8);
        if(id_x == 0 && id_y == 0)
            printf(" \n %d %d %d %d %d %d", dstIdx_b, srcIdx_b, x_b, y_b, (int)srcPtr[srcIdx_b], (int)dstPtr[dstIdx_b]);
    }
}

// template <typename T>
// __global__ void brightness_pkd3_pln3_tensor(T *srcPtr,
//                                       uint2 srcStridesNH,
//                                       T *dstPtr,
//                                       uint3 dstStridesNCH,
//                                       int channelDst,
//                                       unsigned int *x_offset_r,
//                                       unsigned int *y_offset_r,
//                                       unsigned int *x_offset_g,
//                                       unsigned int *y_offset_g,
//                                       unsigned int *x_offset_b,
//                                       unsigned int *y_offset_b,
//                                       RpptROIPtr roiTensorPtrSrc)
// {
//     int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
//     int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
//     int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

//     if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         return;
//     }
    
//     int x_r, y_r, x_g, y_g, x_b, y_b;

//     x_r = id_x + x_offset_r[id_z];
//     y_r = id_y + y_offset_r[id_z];

//     x_g = id_x + x_offset_g[id_z];
//     y_g = id_y + y_offset_g[id_z];

//     x_b = id_x + x_offset_b[id_z];
//     y_b = id_y + y_offset_b[id_z];
    
//     uint srcIdx = (id_z * srcStridesNH.x) + ((id_y + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((id_x + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//     uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
//     uint dstIdx_r = dstIdx;
//     uint dstIdx_g = dstIdx_r + dstStridesNCH.y;
//     uint dstIdx_b = dstIdx_g + dstStridesNCH.y;

//     d_float24 src_f24, dst_f24;
//     d_float8 srcR_f24, srcG_f24, srcB_f24;
//     d_float8 dstR_f8, dstG_f8, dstB_f8;

//     rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
//     rpp_hip_pack_float24_pln3_and_store24_pln3(dstPtr + dstIdx, dstStridesNCH.y, &dst_f24);
    
//     if((y_r >= 0) && (y_r <  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r < roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         srcIdx_r = (id_z * srcStridesNCH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3);
//         rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &src_f24);
//         rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &srcR_f8);
//     }
    
//     if((y_g >= 0) && (y_g <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         srcIdx_g = (id_z * srcStridesNCH.x) + ((y_g + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNH.y) + ((x_g + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3) + 1;
//         rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_g, &srcG_f8);
//         rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_g , &srcG_f8);
//     }

//     if((y_b >= 0) && (y_b <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
//     {
//         srcIdx_b = (id_z * srcStridesNCH.x) + ((y_b + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + ((x_b + roiTensorPtrSrc[id_z].xywhROI.xy.x) * 3) + 2;
//         rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx_b, &srcB_f8);
//         rpp_hip_pack_float8_and_store8(dstPtr + dstIdx_b, &srcB_f8);
//     }
    
// }

template <typename T>
__global__ void glitch_pln3_pkd3_tensor(T *srcPtr,
                                      uint3 srcStridesNCH,
                                      T *dstPtr,
                                      uint2 dstStridesNH,
                                      int channelDst,
                                      unsigned int *x_offset_r,
                                      unsigned int *y_offset_r,
                                      unsigned int *x_offset_g,
                                      unsigned int *y_offset_g,
                                      unsigned int *x_offset_b,
                                      unsigned int *y_offset_b,
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
    d_float24 pix_f24, pix_temp;
    rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx, srcStridesNCH.y, &pix_f24);
    //rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);

    int x_r, y_r, x_g, y_g, x_b, y_b;

    x_r = id_x + x_offset_r[id_z];
    y_r = id_y + y_offset_r[id_z];

    x_g = id_x + x_offset_g[id_z];
    y_g = id_y + y_offset_g[id_z];

    x_b = id_x + x_offset_b[id_z];
    y_b = id_y + y_offset_b[id_z];

    if((y_r >= 0) && (y_r <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_r >= 0) && (x_r <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdx_r = (id_z * srcStridesNCH.x) + ((y_r + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_r + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx_r, srcStridesNCH.y, &pix_temp);
        pix_f24.f4[0] = pix_temp.f4[0];
        pix_f24.f4[1] = pix_temp.f4[1];
    }
    // if((y_g >= 0) && (y_g <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_g >= 0) && (x_g <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    // {
    //     uint srcIdx_g = srcStridesNCH.y + (id_z * srcStridesNCH.x) + ((y_g + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_g + roiTensorPtrSrc[id_z].xywhROI.xy.x);
    //     rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx_g, srcStridesNCH.y, &pix_temp);
    //     pix_f24.f4[2] = pix_temp.f4[2];
    //     pix_f24.f4[3] = pix_temp.f4[3];
    // }
    if((y_b >= 0) && (y_b <=  roiTensorPtrSrc[id_z].xywhROI.roiHeight) && (x_b >= 0) && (x_b <= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        uint srcIdx_b = (srcStridesNCH.y * 2) + (id_z * srcStridesNCH.x) + ((y_b + roiTensorPtrSrc[id_z].xywhROI.xy.y) * srcStridesNCH.z) + (x_b + roiTensorPtrSrc[id_z].xywhROI.xy.x);
        rpp_hip_load24_pln3_and_unpack_to_float24_pln3(srcPtr + srcIdx_b, srcStridesNCH.y, &pix_temp);
        pix_f24.f4[4] = pix_temp.f4[4];
        pix_f24.f4[5] = pix_temp.f4[5];
    }
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &pix_f24);
}


template <typename T>
RppStatus hip_exec_glitch_tensor(T *srcPtr,
                                     RpptDescPtr srcDescPtr,
                                     T *dstPtr,
                                     RpptDescPtr dstDescPtr,
                                     RpptROIPtr roiTensorPtrSrc,
                                     RpptRoiType roiType,
                                     rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);
    std::cerr << "GLITCH TENSOR\n";
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (dstDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(glitch_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        hipLaunchKernelGGL(glitch_pln3_pkd3_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                           dstDescPtr->c,
                           handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[1].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[2].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[3].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[4].uintmem,
                           handle.GetInitHandle()->mem.mgpu.uintArr[5].uintmem,
                           roiTensorPtrSrc);
    }
    return RPP_SUCCESS;
}