#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus remap_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *rowRemapTable,
                                  Rpp32u *colRemapTable,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    __m128 pSrcChannel = _mm_set1_ps(srcDescPtr->c);
    __m128 pSrcStride = _mm_set1_ps(srcDescPtr->strides.hStride);

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable + (roi.xywhROI.roiWidth * roi.xywhROI.roiHeight * batchCount);
        colRemapTableImage = colRemapTable + (roi.xywhROI.roiWidth * roi.xywhROI.roiHeight * batchCount);
        Rpp32u alignedLength = roi.xywhROI.roiWidth & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s vectorIncrement = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32s remapSrcLocArray[4] = {0};     // Since 4 dst pixels are processed per iteration

        // Remap with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);
                colRemapTableTemp = colRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32u remappedSrcLoc = (*rowRemapTableTemp * srcDescPtr->strides.hStride) + (*colRemapTableTemp * srcDescPtr->c);
                    *dstPtrTempR++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTempG++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTempB++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Remap with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);
                colRemapTableTemp = colRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32u remappedSrcLoc = (*rowRemapTableTemp * srcDescPtr->strides.hStride) + *colRemapTableTemp;

                    *dstPtrTemp++ = *(srcPtrRowR + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowG + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrRowB + remappedSrcLoc);
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);
                colRemapTableTemp = colRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m128i pRow;
                    compute_remap_loc(rowRemapTableTemp, colRemapTableTemp, remapSrcLocArray, pSrcStride, pSrcChannel);
                    rpp_simd_load(rpp_nn_load_u8pkd3, srcPtrChannel, remapSrcLocArray, pRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                    rowRemapTableTemp += vectorIncrement;
                    colRemapTableTemp += vectorIncrement;
                }
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp32u remappedSrcLoc = (*rowRemapTableTemp * srcDescPtr->strides.hStride) + (*colRemapTableTemp * srcDescPtr->c);
                    *dstPtrTemp++ = *(srcPtrChannel + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrChannel + 1 + remappedSrcLoc);
                    *dstPtrTemp++ = *(srcPtrChannel + 2 + remappedSrcLoc);
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);
                colRemapTableTemp = colRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8u * dstPtrTempChannel = dstPtrTemp;
                    Rpp8u * srcPtrTempChannel = srcPtrChannel;
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        *dstPtrTempChannel = *(srcPtrTempChannel + (*rowRemapTableTemp * srcDescPtr->strides.hStride) + *colRemapTableTemp);
                        dstPtrTempChannel += dstDescPtr->strides.cStride;
                        srcPtrTempChannel += srcDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}