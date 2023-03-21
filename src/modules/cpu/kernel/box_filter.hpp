#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus box_filter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp8u *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32u kernelSize,
                                       RpptROIPtr roiTensorPtrSrc,
                                       RpptRoiType roiType,
                                       RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u padLength = kernelSize / 2;
        Rpp32u kernelSizeMinusOne = kernelSize - 1;
        Rpp32f kernelSizeInverseSquare = 1.0 / (kernelSize * kernelSize);
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u precomputedRowsIncrement = kernelSizeMinusOne * srcDescPtr->strides.hStride;
        Rpp16s convolutionFactor = (Rpp16s) std::ceil(65536 * kernelSizeInverseSquare);
        __m128i pxMul = _mm_set1_epi16(convolutionFactor);

        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = bufferLength & ~15;

            // Loop for each channel
            for(int c = 0; c < srcDescPtr->c; c++)
            {
                Rpp8u *srcPtrCol, *dstPtrCol;
                srcPtrCol = srcPtrChannel - (padLength * srcDescPtr->strides.hStride) - padLength;
                dstPtrCol = dstPtrChannel;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i pxScratch0[9], pxScratch1[9];

                    Rpp8u *srcPtrRow, *dstPtrRow;
                    srcPtrRow = srcPtrCol;
                    dstPtrRow = dstPtrCol;

                    // Computation for first destination row
                    Rpp8u *srcPtrRowConv;
                    srcPtrRowConv = srcPtrRow;

                    __m128i pxSrc[2];
                    __m128i pxZero = _mm_setzero_si128();
                    __m128i pxRowConv[2];
                    __m128i pxColConv[2];
                    pxRowConv[0] = pxZero;
                    pxRowConv[1] = pxZero;
                    pxColConv[0] = pxZero;
                    pxColConv[1] = pxZero;

                    // Convolution execution
                    // Convolution width loop for first row
                    for (int kWidth = 0; kWidth < kernelSize; kWidth++)
                    {
                        pxSrc[0] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + kWidth));
                        pxSrc[1] = _mm_unpackhi_epi8(pxSrc[0], pxZero);
                        pxSrc[0] = _mm_unpacklo_epi8(pxSrc[0], pxZero);
                        pxColConv[0] = _mm_add_epi16(pxColConv[0], pxSrc[0]);
                        pxColConv[1] = _mm_add_epi16(pxColConv[1], pxSrc[1]);
                    }
                    pxRowConv[0] = pxColConv[0];
                    pxRowConv[1] = pxColConv[1];
                    srcPtrRowConv += srcDescPtr->strides.hStride;

                    // Convolution height loop for remaining rows in kernelSize
                    for (int kHeight = 0; kHeight < kernelSizeMinusOne; kHeight++)
                    {
                        // Reset Column Convolution Result Register
                        pxColConv[0] = pxZero;
                        pxColConv[1] = pxZero;

                        // Convolution width loop for each row
                        for (int kWidth = 0; kWidth < kernelSize; kWidth++)
                        {
                            pxSrc[0] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + kWidth));
                            pxSrc[1] = _mm_unpackhi_epi8(pxSrc[0], pxZero);
                            pxSrc[0] = _mm_unpacklo_epi8(pxSrc[0], pxZero);
                            pxColConv[0] = _mm_add_epi16(pxColConv[0], pxSrc[0]);
                            pxColConv[1] = _mm_add_epi16(pxColConv[1], pxSrc[1]);
                        }
                        pxScratch0[kHeight] = pxColConv[0];
                        pxScratch1[kHeight] = pxColConv[1];

                        pxRowConv[0] = _mm_add_epi16(pxColConv[0], pxRowConv[0]);
                        pxRowConv[1] = _mm_add_epi16(pxColConv[1], pxRowConv[1]);
                        srcPtrRowConv += srcDescPtr->strides.hStride;
                    }

                    // Multiply by convolution factor and write to destination
                    pxRowConv[0] = _mm_mulhi_epi16(pxRowConv[0], pxMul);
                    pxRowConv[1] = _mm_mulhi_epi16(pxRowConv[1], pxMul);
                    pxRowConv[0] = _mm_packus_epi16(pxRowConv[0], pxRowConv[1]);
                    _mm_storeu_si128((__m128i *)dstPtrRow, pxRowConv[0]);

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;

                    // Computation for remaining destination rows
                    for(int i = 0; i < roi.xywhROI.roiHeight - 1; i++)
                    {
                        Rpp8u *srcPtrRowConv;
                        srcPtrRowConv = srcPtrRow;

                        pxRowConv[0] = pxZero;
                        pxRowConv[1] = pxZero;
                        pxColConv[0] = pxZero;
                        pxColConv[1] = pxZero;

                        // Convolution execution
                        // Convolution height loop for pre-computed rows
                        for (int kHeight = 0; kHeight < kernelSizeMinusOne; kHeight++)
                        {
                            pxRowConv[0] = _mm_add_epi16(pxRowConv[0], pxScratch0[kHeight]);
                            pxRowConv[1] = _mm_add_epi16(pxRowConv[1], pxScratch1[kHeight]);
                        }
                        srcPtrRowConv += precomputedRowsIncrement;

                        // Convolution width loop for last row
                        for (int kWidth = 0; kWidth < kernelSize; kWidth++)
                        {
                            pxSrc[0] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + kWidth));
                            pxSrc[1] = _mm_unpackhi_epi8(pxSrc[0], pxZero);
                            pxSrc[0] = _mm_unpacklo_epi8(pxSrc[0], pxZero);
                            pxColConv[0] = _mm_add_epi16(pxColConv[0], pxSrc[0]);
                            pxColConv[1] = _mm_add_epi16(pxColConv[1], pxSrc[1]);
                        }

                        pxScratch0[kernelSizeMinusOne] = pxColConv[0];
                        pxScratch1[kernelSizeMinusOne] = pxColConv[1];

                        pxRowConv[0] = _mm_add_epi16(pxColConv[0], pxRowConv[0]);
                        pxRowConv[1] = _mm_add_epi16(pxColConv[1], pxRowConv[1]);

                        // Multiply by convolution factor and write to destination
                        pxRowConv[0] = _mm_mulhi_epi16(pxRowConv[0], pxMul);
                        pxRowConv[1] = _mm_mulhi_epi16(pxRowConv[1], pxMul);
                        pxRowConv[0] = _mm_packus_epi16(pxRowConv[0], pxRowConv[1]);
                        _mm_storeu_si128((__m128i *)dstPtrRow, pxRowConv[0]);

                        for (int kHeight = 0; kHeight < kernelSizeMinusOne; kHeight++)
                        {
                            pxScratch0[kHeight] = pxScratch0[kHeight + 1];
                            pxScratch1[kHeight] = pxScratch1[kHeight + 1];
                        }

                        srcPtrRow += srcDescPtr->strides.hStride;
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }

                    srcPtrCol += 16;
                    dstPtrCol += 16;
                }

                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}
