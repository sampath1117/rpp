/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

template<typename T>
RppStatus slice_host_tensor(T *srcPtr,
                            RpptGenericDescPtr srcGenericDescPtr,
                            T *dstPtr,
                            RpptGenericDescPtr dstGenericDescPtr,
                            Rpp32s *anchorTensor,
                            Rpp32s *shapeTensor,
                            T* fillValue,
                            bool enablePadding,
                            RpptROI3DPtr roiGenericPtrSrc,
                            RpptRoi3DType roiType,
                            RppLayoutParams layoutParams,
                            rpp::Handle& handle)
{
    RpptROI3D roiDefault;
    if(srcGenericDescPtr->layout==RpptLayout::NCDHW)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[4], (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2]};
    else if(srcGenericDescPtr->layout==RpptLayout::NDHWC)
        roiDefault = {0, 0, 0, (Rpp32s)srcGenericDescPtr->dims[3], (Rpp32s)srcGenericDescPtr->dims[2], (Rpp32s)srcGenericDescPtr->dims[1]};
    Rpp32u numThreads = handle.GetNumThreads();
    Rpp32u numDims = srcGenericDescPtr->numDims - 1;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstGenericDescPtr->dims[0]; batchCount++)
    {
        RpptROI3D roi;
        RpptROI3DPtr roiPtrInput = &roiGenericPtrSrc[batchCount];
        compute_roi3D_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcGenericDescPtr->strides[0];
        dstPtrImage = dstPtr + batchCount * dstGenericDescPtr->strides[0];

        T *srcPtrChannel, *dstPtrChannel;
        dstPtrChannel = dstPtrImage;

        Rpp32s *anchor = &anchorTensor[batchCount * numDims];
        Rpp32s *shape = &shapeTensor[batchCount * numDims];

        // Slice without fused output-layout toggle (NCDHW -> NCDHW)
        if((srcGenericDescPtr->layout == RpptLayout::NCDHW) && (dstGenericDescPtr->layout == RpptLayout::NCDHW))
        {
            srcPtrChannel = srcPtrImage + (anchor[1] * srcGenericDescPtr->strides[2]) + (anchor[2] * srcGenericDescPtr->strides[3]) + (anchor[3] * layoutParams.bufferMultiplier);
            Rpp32u maxDepth = std::min(shape[1], roi.xyzwhdROI.roiDepth - anchor[1]);
            Rpp32u maxHeight = std::min(shape[2], roi.xyzwhdROI.roiHeight - anchor[2]);
            Rpp32u maxWidth = std::min(shape[3], roi.xyzwhdROI.roiWidth - anchor[3]);
            Rpp32u bufferLength = maxWidth * layoutParams.bufferMultiplier;
            Rpp32u copyLengthInBytes = bufferLength * sizeof(T);

            // if padding is required, fill the buffer with fill value specified
            bool needPadding = (((anchor[1] + shape[1]) > roi.xyzwhdROI.roiDepth) ||
                                ((anchor[2] + shape[2]) > roi.xyzwhdROI.roiHeight) ||
                                ((anchor[3] + shape[3]) > roi.xyzwhdROI.roiWidth));
            if(needPadding && enablePadding)
                std::fill(dstPtrImage, dstPtrImage + dstGenericDescPtr->strides[0] - 1, *fillValue);

            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                T *srcPtrDepth, *dstPtrDepth;
                srcPtrDepth = srcPtrChannel;
                dstPtrDepth = dstPtrChannel;

                for(int i = 0; i < maxDepth; i++)
                {
                    T *srcPtrRow, *dstPtrRow;
                    srcPtrRow = srcPtrDepth;
                    dstPtrRow = dstPtrDepth;

                    for(int j = 0; j < maxHeight; j++)
                    {
                        memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                        srcPtrRow += srcGenericDescPtr->strides[3];
                        dstPtrRow += dstGenericDescPtr->strides[3];
                    }
                    srcPtrDepth += srcGenericDescPtr->strides[2];
                    dstPtrDepth += dstGenericDescPtr->strides[2];
                }
                srcPtrChannel += srcGenericDescPtr->strides[1];
                dstPtrChannel += srcGenericDescPtr->strides[1];
            }
        }

        // Slice without fused output-layout toggle (NDHWC -> NDHWC)
        else if((srcGenericDescPtr->layout == RpptLayout::NDHWC) && (dstGenericDescPtr->layout == RpptLayout::NDHWC))
        {
            srcPtrChannel = srcPtrImage + (anchor[0] * srcGenericDescPtr->strides[1]) + (anchor[1] * srcGenericDescPtr->strides[2]) + (anchor[2] * layoutParams.bufferMultiplier);
            Rpp32u maxDepth = std::min(shape[0], roi.xyzwhdROI.roiDepth - anchor[0]);
            Rpp32u maxHeight = std::min(shape[1], roi.xyzwhdROI.roiHeight - anchor[1]);
            Rpp32u maxWidth = std::min(shape[2], roi.xyzwhdROI.roiWidth - anchor[2]);
            Rpp32u bufferLength = maxWidth * layoutParams.bufferMultiplier;
            Rpp32u copyLengthInBytes = bufferLength * sizeof(T);

            // if padding is required, fill the buffer with fill value specified
            bool needPadding = (((anchor[0] + shape[0]) > roi.xyzwhdROI.roiDepth) ||
                                ((anchor[1] + shape[1]) > roi.xyzwhdROI.roiHeight) ||
                                ((anchor[2] + shape[2]) > roi.xyzwhdROI.roiWidth));
            if(needPadding && enablePadding)
                std::fill(dstPtrImage, dstPtrImage + dstGenericDescPtr->strides[0] - 1, *fillValue);

            T *srcPtrDepth = srcPtrChannel;
            T *dstPtrDepth = dstPtrChannel;
            for(int i = 0; i < maxDepth; i++)
            {
                T *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrDepth;
                dstPtrRow = dstPtrDepth;
                for(int j = 0; j < maxHeight; j++)
                {
                    memcpy(dstPtrRow, srcPtrRow, copyLengthInBytes);
                    srcPtrRow += srcGenericDescPtr->strides[2];
                    dstPtrRow += dstGenericDescPtr->strides[2];
                }
                srcPtrDepth += srcGenericDescPtr->strides[1];
                dstPtrDepth += dstGenericDescPtr->strides[1];
            }
        }
    }

    return RPP_SUCCESS;
}
