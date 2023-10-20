#include <hip/hip_runtime.h>
#include <omp.h>
#include "rpp_hip_common.hpp"

template <typename T>
__global__ void slice_voxel_ncdhw_tensor(T *srcPtr,
                                         uint3 srcStridesCDH,
                                         T *dstPtr,
                                         uint3 dstStridesCDH,
                                         int channels,
                                         uint3 validShapeDHW,
                                         int *anchor,
                                         int *shape,
                                         RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= validShapeDHW.x) || (id_y >= validShapeDHW.y) || (id_x >= validShapeDHW.z))
    {
        return;
    }

    uint srcIdx = ((id_z + anchor[1]) * srcStridesCDH.y) + ((id_y + anchor[2]) * srcStridesCDH.z) + (id_x + anchor[3]);
    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;

    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCDH.x;
        dstIdx += dstStridesCDH.x;
    }
}

template <typename T>
__global__ void fill_value_ncdhw_tensor(T *dstPtr,
                                        uint3 dstStridesCDH,
                                        int channels,
                                        uint3 dstDimsDHW,
                                        T *fillValue)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= dstDimsDHW.x) || (id_y >= dstDimsDHW.y) || (id_x >= dstDimsDHW.z))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesCDH.y) + (id_y * dstStridesCDH.z) + id_x;
    d_float8 val_f8;
    val_f8.f4[0] = (float4)(*fillValue);
    val_f8.f4[1] = val_f8.f4[0];
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        dstIdx += dstStridesCDH.x;
    }
}

template <typename T>
__global__ void slice_voxel_ndhwc_tensor(T *srcPtr,
                                         uint2 srcStridesDH,
                                         T *dstPtr,
                                         uint2 dstStridesDH,
                                         uint3 validShapeDHW,
                                         int *anchor,
                                         int *shape,
                                         RpptROI3DPtr roiGenericPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= validShapeDHW.x) || (id_y >= validShapeDHW.y) || (id_x >= validShapeDHW.z))
    {
        return;
    }

    uint srcIdx = ((id_z + anchor[0]) * srcStridesDH.x) + ((id_y + anchor[1]) * srcStridesDH.y) + (id_x + anchor[2]) * 3;
    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;

    d_float24 val_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &val_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

template <typename T>
__global__ void fill_value_ndhwc_tensor(T *dstPtr,
                                        uint2 dstStridesDH,
                                        uint3 dstDimsDHW,
                                        T *fillValue)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;              // D - outer most dim

    if ((id_z >= dstDimsDHW.x) || (id_y >= dstDimsDHW.y) || (id_x >= dstDimsDHW.z))
    {
        return;
    }

    uint dstIdx = (id_z * dstStridesDH.x) + (id_y * dstStridesDH.y) + id_x * 3;
    d_float24 val_f24;
    val_f24.f4[0] = (float4)(*fillValue);
    val_f24.f4[1] = val_f24.f4[0];
    val_f24.f4[2] = val_f24.f4[0];
    val_f24.f4[3] = val_f24.f4[0];
    val_f24.f4[4] = val_f24.f4[0];
    val_f24.f4[5] = val_f24.f4[0];
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

template <typename T>
RppStatus hip_exec_slice_voxel_tensor(T *srcPtr,
                                      RpptGenericDescPtr srcGenericDescPtr,
                                      T *dstPtr,
                                      RpptGenericDescPtr dstGenericDescPtr,
                                      Rpp32s *anchorTensor,
                                      Rpp32s *shapeTensor,
                                      T *fillValue,
                                      bool enablePadding,
                                      RpptROI3DPtr roiGenericPtrSrc,
                                      rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr->numDims - 1;
    // create a kernel for filling padded region with value
    if (dstGenericDescPtr->layout == RpptLayout::NCDHW)
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstGenericDescPtr->strides[3] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[3];               // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[2];               // D - depth (z direction)

        if(enablePadding)
        {
            for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
            {
                Rpp32s *anchor = &anchorTensor[batchCount * numDims];
                Rpp32s *shape = &shapeTensor[batchCount * numDims];
                RpptROI3DPtr roi = &roiGenericPtrSrc[batchCount];
                Rpp32u maxDepth = std::min(shape[1], roi->xyzwhdROI.roiDepth - anchor[1]);
                Rpp32u maxHeight = std::min(shape[2], roi->xyzwhdROI.roiHeight - anchor[2]);
                Rpp32u maxWidth = std::min(shape[3], roi->xyzwhdROI.roiWidth - anchor[3]);

                // check if padding is needed
                bool needPadding = (((anchor[1] + shape[1]) > roi->xyzwhdROI.roiDepth) ||
                                    ((anchor[2] + shape[2]) > roi->xyzwhdROI.roiHeight) ||
                                    ((anchor[3] + shape[3]) > roi->xyzwhdROI.roiWidth));

                // launch kernel for filling the padded region with fill value specified
                if(needPadding)
                {
                    hipLaunchKernelGGL(fill_value_ncdhw_tensor,
                                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                       dim3(localThreads_x, localThreads_y, localThreads_z),
                                       0,
                                       handle.GetStream(),
                                       dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                       make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                                       dstGenericDescPtr->dims[1],
                                       make_uint3(shape[1], shape[2], shape[3]),
                                       fillValue);
                }
            }
        }
        hipStreamSynchronize(handle.GetStream());

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            RpptROI3DPtr roi = &roiGenericPtrSrc[batchCount];
            Rpp32u maxDepth = std::min(shape[1], roi->xyzwhdROI.roiDepth - anchor[1]);
            Rpp32u maxHeight = std::min(shape[2], roi->xyzwhdROI.roiHeight - anchor[2]);
            Rpp32u maxWidth = std::min(shape[3], roi->xyzwhdROI.roiWidth - anchor[3]);

            hipLaunchKernelGGL(slice_voxel_ncdhw_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint3(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2], srcGenericDescPtr->strides[3]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint3(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2], dstGenericDescPtr->strides[3]),
                               dstGenericDescPtr->dims[1],
                               make_uint3(maxDepth, maxHeight, maxWidth),
                               anchor,
                               shape,
                               &roiGenericPtrSrc[batchCount]);
        }
    }
    else if (dstGenericDescPtr->layout == RpptLayout::NDHWC)
    {
        int localThreads_x = LOCAL_THREADS_X;
        int localThreads_y = LOCAL_THREADS_Y;
        int localThreads_z = LOCAL_THREADS_Z;
        int globalThreads_x = (dstGenericDescPtr->strides[2] / 3 + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[2];               // H - height (y direction)
        int globalThreads_z = dstGenericDescPtr->dims[1];               // D - depth (z direction)


        if(enablePadding)
        {
            for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
            {
                Rpp32s *anchor = &anchorTensor[batchCount * numDims];
                Rpp32s *shape = &shapeTensor[batchCount * numDims];
                RpptROI3DPtr roi = &roiGenericPtrSrc[batchCount];
                Rpp32u maxDepth = std::min(shape[0], roi->xyzwhdROI.roiDepth - anchor[0]);
                Rpp32u maxHeight = std::min(shape[1], roi->xyzwhdROI.roiHeight - anchor[1]);
                Rpp32u maxWidth = std::min(shape[2], roi->xyzwhdROI.roiWidth - anchor[2]);

                // check if padding is needed
                bool needPadding = (((anchor[0] + shape[0]) > roi->xyzwhdROI.roiDepth) ||
                                    ((anchor[1] + shape[1]) > roi->xyzwhdROI.roiHeight) ||
                                    ((anchor[2] + shape[2]) > roi->xyzwhdROI.roiWidth));

                // launch kernel for filling the padded region with fill value specified
                if(needPadding)
                {
                    hipLaunchKernelGGL(fill_value_ndhwc_tensor,
                                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                                       dim3(localThreads_x, localThreads_y, localThreads_z),
                                       0,
                                       handle.GetStream(),
                                       dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                       make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                       make_uint3(shape[0], shape[1], shape[2]),
                                       fillValue);
                }
            }
        }
        hipStreamSynchronize(handle.GetStream());

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            RpptROI3DPtr roi = &roiGenericPtrSrc[batchCount];
            Rpp32u maxDepth = std::min(shape[0], roi->xyzwhdROI.roiDepth - anchor[0]);
            Rpp32u maxHeight = std::min(shape[1], roi->xyzwhdROI.roiHeight - anchor[1]);
            Rpp32u maxWidth = std::min(shape[2], roi->xyzwhdROI.roiWidth - anchor[2]);

            hipLaunchKernelGGL(slice_voxel_ndhwc_tensor,
                               dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                               dim3(localThreads_x, localThreads_y, localThreads_z),
                               0,
                               handle.GetStream(),
                               srcPtr + (batchCount * srcGenericDescPtr->strides[0]),
                               make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                               dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                               make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                               make_uint3(maxDepth, maxHeight, maxWidth),
                               anchor,
                               shape,
                               &roiGenericPtrSrc[batchCount]);
        }
    }

    return RPP_SUCCESS;
}