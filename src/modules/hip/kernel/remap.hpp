#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"


// -------------------- Set 2 - Nearest Neighbor Interpolation --------------------

template <typename T>
__global__ void remap_pln_tensor(T *srcPtr,
                                uint3 srcStridesNCH,
                                T *dstPtr,
                                uint3 dstStridesNCH,
                                uint2 dstDimsWH,
                                int channelsDst,
                                uint *rowRemapTable,
                                uint *colRemapTable,
                                RpptROIPtr roiTensorPtrSrc)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= dstDimsWH.y) || (id_x >= dstDimsWH.x))
    {
        return;
    }

    uint srcIdx = (id_z * srcStridesNCH.x);
    uint dstIdx = (id_z * dstStridesNCH.x) + (id_y * dstStridesNCH.z) + id_x;
    
    int4 srcRoi_i4 = *(int4 *)&roiTensorPtrSrc[id_z];

    uint rowRemapVal =  *(rowRemapTable + (id_z * srcRoi_i4.z * srcRoi_i4.w) + (id_y * srcRoi_i4.z) + id_x);
    uint colRemapVal =  *(colRemapTable + (id_z * srcRoi_i4.z * srcRoi_i4.w) + (id_y * srcRoi_i4.z) + id_x);
    uint srcRemapLoc =  (rowRemapVal * dstStridesNCH.z) + colRemapVal;
    
    dstPtr[dstIdx] = srcPtr[srcRemapLoc];

    // if (channelsDst == 3)
    // {
    //     srcIdx += srcStridesNCH.y;
    //     dstIdx += dstStridesNCH.y;

    //     rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);

    //     srcIdx += srcStridesNCH.y;
    //     dstIdx += dstStridesNCH.y;

    //     rpp_hip_interpolate8_nearest_neighbor_pln1(srcPtr + srcIdx, srcStridesNCH.z, &locSrc_f16, &srcRoi_i4, &dst_f8);
    //     rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &dst_f8);
    // }
}
// -------------------- Set 3 - Kernel Executors --------------------

template <typename T>
RppStatus hip_exec_remap_tensor(T *srcPtr,
                                RpptDescPtr srcDescPtr,
                                T *dstPtr,
                                RpptDescPtr dstDescPtr,
                                Rpp32u *rowRemapTable,
                                Rpp32u *colRemapTable,
                                RpptROIPtr roiTensorPtrSrc,
                                RpptRoiType roiType,
                                rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = dstDescPtr->strides.hStride;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();
    
    if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
    {
        hipLaunchKernelGGL(remap_pln_tensor,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.cStride, srcDescPtr->strides.hStride),
                           dstPtr,
                           make_uint3(dstDescPtr->strides.nStride, dstDescPtr->strides.cStride, dstDescPtr->strides.hStride),
                           make_uint2(dstDescPtr->w, dstDescPtr->h),
                           dstDescPtr->c,
                           rowRemapTable,
                           colRemapTable,
                           roiTensorPtrSrc);
    }
}