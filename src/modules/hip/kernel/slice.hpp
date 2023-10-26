#include <hip/hip_runtime.h>
#include <omp.h>
#include "rpp_hip_common.hpp"

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
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

template <typename T>
__global__ void fill_value_nchw_tensor(T *dstPtr,
                                       uint2 dstStridesCH,
                                       int channels,
                                       uint2 dstDimsHW,
                                       T *fillValue)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner

    if ((id_y >= dstDimsHW.x) || (id_x >= dstDimsHW.y))
    {
        return;
    }

    uint dstIdx = (id_y * dstStridesCH.y) + id_x;
    d_float8 val_f8;
    val_f8.f4[0] = (float4)(*fillValue);
    val_f8.f4[1] = val_f8.f4[0];
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        dstIdx += dstStridesCH.x;
    }
}

template <typename T>
__global__ void fill_value_nhwc_tensor(T *dstPtr,
                                       uint dstStridesH,
                                       int channels,
                                       uint2 dstDimsHW,
                                       T *fillValue)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner

    if ((id_y >= dstDimsHW.x) || (id_x >= dstDimsHW.y))
    {
        return;
    }

    uint dstIdx = (id_y * dstStridesH) + id_x * 3;
    d_float24 val_f24;
    val_f24.f4[0] = (float4)(*fillValue);
    val_f24.f4[1] = val_f24.f4[0];
    val_f24.f4[2] = val_f24.f4[0];
    val_f24.f4[3] = val_f24.f4[0];
    val_f24.f4[4] = val_f24.f4[0];
    val_f24.f4[5] = val_f24.f4[0];
    rpp_hip_pack_float24_pkd3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);

}

template <typename T>
__global__ void slice_ncdhw_hip_tensor(T *srcPtr,
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
__global__ void slice_ndhwc_hip_tensor(T *srcPtr,
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
__global__ void slice_nchw_hip_tensor(T *srcPtr,
                                      uint2 srcStridesCH,
                                      T *dstPtr,
                                      uint2 dstStridesCH,
                                      int channels,
                                      uint2 validShapeHW)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // W - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner

    if ((id_y >= validShapeHW.x) || (id_x >= validShapeHW.y))
    {
        return;
    }

    uint srcIdx = (id_y * srcStridesCH.y) + id_x;
    uint dstIdx = (id_y * dstStridesCH.y) + id_x;

    d_float8 val_f8;
    for(int c = 0; c < channels; c++)
    {
        rpp_hip_load8_and_unpack_to_float8(srcPtr + srcIdx, &val_f8);
        rpp_hip_pack_float8_and_store8(dstPtr + dstIdx, &val_f8);
        srcIdx += srcStridesCH.x;
        dstIdx += dstStridesCH.x;
    }
}

template <typename T>
__global__ void slice_nhwc_hip_tensor(T *srcPtr,
                                      uint srcStridesH,
                                      T *dstPtr,
                                      uint dstStridesH,
                                      uint2 validShapeHW)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;        // WC - inner most dim vectorized
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;              // H - second to inner

    if ((id_y >= validShapeHW.x) || (id_x >= validShapeHW.y))
    {
        return;
    }

    uint srcIdx = (id_y * srcStridesH) + id_x * 3;
    uint dstIdx = (id_y * dstStridesH) + id_x * 3;

    d_float24 val_f24;
    rpp_hip_load24_pkd3_and_unpack_to_float24_pln3(srcPtr + srcIdx, &val_f24);
    rpp_hip_pack_float24_pln3_and_store24_pkd3(dstPtr + dstIdx, &val_f24);
}

template <typename T>
RppStatus hip_exec_slice_tensor(T *srcPtr,
                                RpptGenericDescPtr srcGenericDescPtr,
                                T *dstPtr,
                                RpptGenericDescPtr dstGenericDescPtr,
                                Rpp32s *anchorTensor,
                                Rpp32s *shapeTensor,
                                T *fillValue,
                                bool enablePadding,
                                Rpp32u *roiTensor,
                                rpp::Handle& handle)
{
    Rpp32u numDims = srcGenericDescPtr->numDims - 1;
    if(numDims == 3)
    {
        // create a kernel for filling padded region with fill value specified
        if (dstGenericDescPtr->layout == RpptLayout::NCHW)
        {
            int globalThreads_x = (dstGenericDescPtr->strides[2] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
            int globalThreads_y = dstGenericDescPtr->dims[2];               // H - height (y direction)
            int globalThreads_z = 1;

            if(enablePadding)
            {
                for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
                {
                    Rpp32s *anchor = &anchorTensor[batchCount * numDims];
                    Rpp32s *shape = &shapeTensor[batchCount * numDims];
                    Rpp32u *roi = roiTensor + batchCount * numDims * 2;
                    Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
                    Rpp32u maxHeight = std::min(shape[1], length[1] - anchor[1]);
                    Rpp32u maxWidth = std::min(shape[2], length[2] - anchor[2]);

                    // check if padding is needed
                    bool needPadding = (((anchor[1] + shape[1]) > length[1]) ||
                                        ((anchor[2] + shape[2]) > length[2]));

                    // launch kernel for filling the padded region with fill value specified
                    if(needPadding)
                    {
                        hipLaunchKernelGGL(fill_value_nchw_tensor,
                                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                           0,
                                           handle.GetStream(),
                                           dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                           make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                           dstGenericDescPtr->dims[1],
                                           make_uint2(shape[1], shape[2]),
                                           fillValue);
                    }
                }
            }
            hipStreamSynchronize(handle.GetStream());

            for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
            {
                Rpp32s *anchor = &anchorTensor[batchCount * numDims];
                Rpp32s *shape = &shapeTensor[batchCount * numDims];
                Rpp32u *roi = roiTensor + batchCount * numDims * 2;
                Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
                Rpp32u maxHeight = std::min(shape[1], length[1] - anchor[1]);
                Rpp32u maxWidth = std::min(shape[2], length[2] - anchor[2]);
                T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[1] * srcGenericDescPtr->strides[2] + anchor[0];
                T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);

                hipLaunchKernelGGL(slice_nchw_hip_tensor,
                                   dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                   dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                   0,
                                   handle.GetStream(),
                                   srcPtrTemp,
                                   make_uint2(srcGenericDescPtr->strides[1], srcGenericDescPtr->strides[2]),
                                   dstPtrTemp,
                                   make_uint2(dstGenericDescPtr->strides[1], dstGenericDescPtr->strides[2]),
                                   dstGenericDescPtr->dims[1],
                                   make_uint2(maxHeight, maxWidth));
            }
        }
    }
    else if(numDims == 2)
    {
        // NHW
        int globalThreads_x = (dstGenericDescPtr->strides[1] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = dstGenericDescPtr->dims[1];               // H - height (y direction)
        int globalThreads_z = 1;

        if(enablePadding)
        {
            for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
            {
                Rpp32s *anchor = &anchorTensor[batchCount * numDims];
                Rpp32s *shape = &shapeTensor[batchCount * numDims];
                Rpp32u *roi = roiTensor + batchCount * numDims * 2;
                Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
                Rpp32u maxHeight = std::min(shape[1], length[1] - anchor[1]);
                Rpp32u maxWidth = std::min(shape[2], length[2] - anchor[2]);

                // check if padding is needed
                bool needPadding = (((anchor[1] + shape[1]) > length[1]) ||
                                    ((anchor[2] + shape[2]) > length[2]));

                // launch kernel for filling the padded region with fill value specified
                if(needPadding)
                {
                    hipLaunchKernelGGL(fill_value_nchw_tensor,
                                        dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                        dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                                        0,
                                        handle.GetStream(),
                                        dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                        make_uint2(0, dstGenericDescPtr->strides[1]),
                                        1,
                                        make_uint2(shape[1], shape[2]),
                                        fillValue);
                }
            }
        }
        hipStreamSynchronize(handle.GetStream());

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxHeight = std::min(shape[1], length[1] - anchor[1]);
            Rpp32u maxWidth = std::min(shape[2], length[2] - anchor[2]);
            T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[1] * srcGenericDescPtr->strides[2] + anchor[0];
            T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);

            hipLaunchKernelGGL(slice_nchw_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, 1),
                               0,
                               handle.GetStream(),
                               srcPtrTemp,
                               make_uint2(0, srcGenericDescPtr->strides[1]),
                               dstPtrTemp,
                               make_uint2(0, dstGenericDescPtr->strides[1]),
                               1,
                               make_uint2(maxHeight, maxWidth));
        }
    }
    else if(numDims == 1)
    {
        int globalThreads_x = (dstGenericDescPtr->strides[0] + 7) >> 3; // W - width (x direction) - vectorized for 8 element loads/stores per channel
        int globalThreads_y = 1;
        int globalThreads_z = 1;

        if(enablePadding)
        {
            for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
            {
                Rpp32s *anchor = &anchorTensor[batchCount * numDims];
                Rpp32s *shape = &shapeTensor[batchCount * numDims];
                Rpp32u *roi = roiTensor + batchCount * numDims * 2;
                Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
                Rpp32u maxLength = std::min(shape[0], length[0] - anchor[0]);

                // check if padding is needed
                bool needPadding = ((anchor[0] + shape[0]) > length[0]);

                // launch kernel for filling the padded region with fill value specified
                if(needPadding)
                {
                    hipLaunchKernelGGL(fill_value_nchw_tensor,
                                       dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                                       dim3(LOCAL_THREADS_X, 1, 1),
                                       0,
                                       handle.GetStream(),
                                       dstPtr + (batchCount * dstGenericDescPtr->strides[0]),
                                       make_uint2(0, 1),
                                       1,
                                       make_uint2(1, shape[0]),
                                       fillValue);
                }
            }
        }
        hipStreamSynchronize(handle.GetStream());

        for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
        {
            Rpp32s *anchor = &anchorTensor[batchCount * numDims];
            Rpp32s *shape = &shapeTensor[batchCount * numDims];
            Rpp32u *roi = roiTensor + batchCount * numDims * 2;
            Rpp32s *length = reinterpret_cast<Rpp32s *>(&roi[numDims]);
            Rpp32u maxLength = std::min(shape[0], length[0] - anchor[0]);
            T *srcPtrTemp = srcPtr + (batchCount * srcGenericDescPtr->strides[0]) + anchor[0];
            T *dstPtrTemp = dstPtr + (batchCount * dstGenericDescPtr->strides[0]);

            hipLaunchKernelGGL(slice_nchw_hip_tensor,
                               dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                               dim3(LOCAL_THREADS_X, 1, 1),
                               0,
                               handle.GetStream(),
                               srcPtrTemp,
                               make_uint2(0, 1),
                               dstPtrTemp,
                               make_uint2(0, 1),
                               1,
                               make_uint2(1, maxLength));
        }
    }

    return RPP_SUCCESS;
}
