#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void rpp_shuffle_and_conv(__m128i *pxSrc, __m128i shuffleMask, __m128i *pxColConv)
{
    // pxSrc[0] - R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 R4 G4 B4 R5
    // pxSrc[1] - R4 G4 B4 R5 G5 B5 R6 G6 B6 R7 G7 B7 R8 G8 B8 R9
    // pxSrc[2] - R8 G8 B8 R9 G9 B9 R10 G10 B10 R11 G11 B11 R12 G12 B12 R13
    // pxSrc[3] - R12 G12 B12 R13 G13 B13 R14 G14 B14 R15 B15 G15 R16 B16 G16 R17
    __m128i pxTemp[4];
    pxTemp[0] = _mm_shuffle_epi8(pxSrc[0], shuffleMask); // X0 0 X1 0 | X2 0 X3 0 | 0 0 0 0 | 0 0 0 0
    pxTemp[1] = _mm_shuffle_epi8(pxSrc[1], shuffleMask); // X4 0 X5 0 | X6 0 X7 0 | 0 0 0 0 | 0 0 0 0
    pxTemp[2] = _mm_shuffle_epi8(pxSrc[2], shuffleMask); // X8 0 X9 0 | X10 0 X11 0 | 0 0 0 0 | 0 0 0 0
    pxTemp[3] = _mm_shuffle_epi8(pxSrc[3], shuffleMask); // X12 0 X13 0 | X14 0 X15 0 | 0 0 0 0 | 0 0 0 0

    pxTemp[0] = _mm_unpacklo_epi64(pxTemp[0], pxTemp[1]); // X0 0 X1 0 | X2 0 X3 0 | X4 0 X5 0 | X6 0 X7 0
    pxTemp[1] = _mm_unpacklo_epi64(pxTemp[2], pxTemp[3]); // X8 0 X9 0 | X10 0 X11 0 | X12 0 X13 0 | X14 0 X15 0

    pxColConv[0] = _mm_add_epi16(pxColConv[0], pxTemp[0]);
    pxColConv[1] = _mm_add_epi16(pxColConv[1], pxTemp[1]);
}

inline void rpp_reset_variables(__m128i *p)
{
    p[0] = xmm_px0;
    p[1] = xmm_px0;
    p[2] = xmm_px0;
    p[3] = xmm_px0;
    p[4] = xmm_px0;
    p[5] = xmm_px0;
}

inline void rpp_pln_to_pkd_lower(__m128i *pxRowConv, __m128i shiftMask, __m128i *pxRes)
{
    // pxRowConv[0] - R0 0 R1 0 R2 0 R3 0 R4 0 R5 0 R6 0 R7 0
    // pxRowConv[2] - G0 0 G1 0 G2 0 G3 0 G4 0 G5 0 G6 0 G7 0
    // xmm_px0      - 0  0 0  0 0  0 0  0 0  0 0  0 0  0 0  0
    // pxRowConv[4] - B0 0 B1 0 B2 0 B3 0 B4 0 B5 0 B6 0 B7 0

    // R0 G0 0 0 | R1 G1 0 0 | R2 G2 0 0 | R3 G3 0 0
    // 0 0 B0 0  | 0 0 B1 0  | 0 0 B2 0 | 0 0 B3 0
    // R0 G0 B0 0 R1 G1 B1 0 R2 G2 B2 0 R3 G3 B3 0
    // R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3 0 0 0 0

    __m128i pTemp[2];
    pTemp[0] = _mm_unpacklo_epi8(pxRowConv[0], pxRowConv[2]);
    pTemp[1] = _mm_unpacklo_epi16(xmm_px0, pxRowConv[4]);
    pTemp[0] = _mm_add_epi8(pTemp[0], pTemp[1]);
    pxRes[0] = _mm_shuffle_epi8(pTemp[0], shiftMask);
}

inline void rpp_pln_to_pkd_higher(__m128i *pxRowConv, __m128i shiftMask, __m128i* pxRes)
{
    // pxRowConv[0] - R0 0 R1 0 | R2 0 R3 0 | R4 0 R5 0 | R6 0 R7 0
    // pxRowConv[2] - G0 0 G1 0 | G2 0 G3 0 | G4 0 G5 0 | G6 0 G7 0
    // xmm_px0      - 0  0 0  0 | 0  0 0  0 | 0  0 0  0 | 0  0 0  0
    // pxRowConv[4] - B0 0 B1 0 | B2 0 B3 0 | B4 0 B5 0 | B6 0 B7 0

    // R4 G4 0 0 | R5 G5 0 0 | R6 G6 0 0 | R7 G7 0 0
    // 0 0 B4 0  | 0 0 B5 0  | 0 0 B6 0 | 0 0 B7 0
    // R4 G4 B4 0 R5 G5 B5 0 R6 G6 B6 0 R7 G7 B7 0
    // R4 G4 B4 R5 G5 B5 R6 G6 B6 R7 G7 B7 0 0 0 0

    __m128i pTemp[2];
    pTemp[0] = _mm_unpackhi_epi8(pxRowConv[0], pxRowConv[2]);
    pTemp[1] = _mm_unpackhi_epi16(xmm_px0, pxRowConv[4]);
    pTemp[0] = _mm_add_epi8(pTemp[0], pTemp[1]);
    pxRes[0] = _mm_shuffle_epi8(pTemp[0], shiftMask);
}

inline void rpp_pln_to_pkd_new(__m128i *pxRowConv, __m128i *pxRes, __m128i pxMask1, __m128i pxMask2)
{
    // Saturate and pack to 8 bits
    pxRowConv[0] = _mm_packus_epi16(pxRowConv[0], pxRowConv[1]); // R0 - R15
    pxRowConv[1] = _mm_packus_epi16(pxRowConv[2], pxRowConv[3]); // G0 - G15
    pxRowConv[2] = _mm_packus_epi16(pxRowConv[4], pxRowConv[5]); // B0 - B15

    // Shuffle to get the outputs
    __m128i pxTemp[4];
    pxTemp[0] = _mm_unpacklo_epi8(pxRowConv[0], pxRowConv[1]); // R0 G0 - R7 G7
    pxTemp[1] = _mm_unpacklo_epi64(pxTemp[0], pxRowConv[2]); // R0 G0 R1 G1 R2 G2 R3 G3 | B0 B1 B2 B3 B4 B5 B6 B7
    pxRes[0] = _mm_shuffle_epi8(pxTemp[1], pxMask1);

    pxTemp[2] = _mm_unpackhi_epi64(pxTemp[0], pxTemp[1]); // R4 G4 R5 G5 R6 G6 R7 G7 | B0 B1 B2 B3 B4 B5 B6 B7
    pxRes[1] = _mm_shuffle_epi8(pxTemp[2], pxMask2);

    pxTemp[0] = _mm_unpackhi_epi8(pxRowConv[0], pxRowConv[1]); // R8 G8 - R15 G15
    pxTemp[1] = _mm_unpackhi_epi64(pxTemp[0], pxRowConv[2]); // R8 G8 R9 G9 R10 G10 R11 G11 | B8 B9 B10 B11 B12 B13 B14 B15
    pxRes[2] = _mm_shuffle_epi8(pxTemp[1], pxMask1);

    pxTemp[2] = _mm_unpackhi_epi64(pxTemp[0], pxTemp[1]); // R12 G12 R13 G13 R14 G14 R15 G15 | B8 B9 B10 B11 B12 B13 B14 B15
    pxRes[3] = _mm_shuffle_epi8(pxTemp[2], pxMask2);
}

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

    const __m128i maskR16 = _mm_setr_epi8(0, 0x80, 3, 0x80, 6, 0x80, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    const __m128i maskG16 = _mm_setr_epi8(1, 0x80, 4, 0x80, 7, 0x80, 10, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
    const __m128i maskB16 = _mm_setr_epi8(2, 0x80, 5, 0x80, 8, 0x80, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

    const __m128i pxMask1 = _mm_setr_epi8(0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11, 0x80, 0x80, 0x80, 0x80);
    const __m128i pxMask2 = _mm_setr_epi8(0, 1, 12, 2, 3, 13, 4, 5, 14, 6, 7, 15, 0x80, 0x80, 0x80, 0x80);
    const __m128i shiftMask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80);

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
                for (; vectorLoopCount < alignedLength; vectorLoopCount += 16)
                {
                    __m128i pxScratch0[kernelSize], pxScratch1[kernelSize];

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
        else if ((srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;

            Rpp8u *srcPtrCol, *dstPtrCol;
            srcPtrCol = srcPtrChannel - (padLength * srcDescPtr->strides.hStride) - padLength * srcDescPtr->c;
            dstPtrCol = dstPtrChannel;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += 48)
            {
                __m128i pxScratchR0[kernelSize], pxScratchR1[kernelSize];
                __m128i pxScratchG0[kernelSize], pxScratchG1[kernelSize];
                __m128i pxScratchB0[kernelSize], pxScratchB1[kernelSize];

                Rpp8u *srcPtrRow, *dstPtrRow;
                srcPtrRow = srcPtrCol;
                dstPtrRow = dstPtrCol;

                // Computation for first destination row
                Rpp8u *srcPtrRowConv;
                srcPtrRowConv = srcPtrRow;

                __m128i pxSrc[4];
                __m128i pxRowConv[6],  pxColConv[6];
                rpp_reset_variables(pxRowConv);
                rpp_reset_variables(pxColConv);

                // Convolution execution
                // Convolution width loop for first row
                for (int kWidth = 0; kWidth < kernelSize; kWidth++)
                {
                    // Load 16R, 16G, 16B values
                    pxSrc[0] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth));
                    pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 12));
                    pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 24));
                    pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 36));

                    rpp_shuffle_and_conv(pxSrc, maskR16, &pxColConv[0]); // R Channel
                    rpp_shuffle_and_conv(pxSrc, maskG16, &pxColConv[2]); // G Channel
                    rpp_shuffle_and_conv(pxSrc, maskB16, &pxColConv[4]); // B Channel
                }

                pxRowConv[0] = pxColConv[0];
                pxRowConv[1] = pxColConv[1];
                pxRowConv[2] = pxColConv[2];
                pxRowConv[3] = pxColConv[3];
                pxRowConv[4] = pxColConv[4];
                pxRowConv[5] = pxColConv[5];
                srcPtrRowConv += srcDescPtr->strides.hStride;

                // Convolution height loop for remaining rows in kernelSize
                for (int kHeight = 0; kHeight < kernelSizeMinusOne; kHeight++)
                {
                    // Reset Column Convolution Result Register
                    rpp_reset_variables(pxColConv);

                    // Convolution width loop for each row
                    for (int kWidth = 0; kWidth < kernelSize; kWidth++)
                    {
                        // Load 16R, 16G, 16B values
                        pxSrc[0] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth));
                        pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 12));
                        pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 24));
                        pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 36));

                        rpp_shuffle_and_conv(pxSrc, maskR16, &pxColConv[0]); // R Channel
                        rpp_shuffle_and_conv(pxSrc, maskG16, &pxColConv[2]); // G Channel
                        rpp_shuffle_and_conv(pxSrc, maskB16, &pxColConv[4]); // B Channel
                    }
                    pxScratchR0[kHeight] = pxColConv[0];
                    pxScratchR1[kHeight] = pxColConv[1];
                    pxScratchG0[kHeight] = pxColConv[2];
                    pxScratchG1[kHeight] = pxColConv[3];
                    pxScratchB0[kHeight] = pxColConv[4];
                    pxScratchB1[kHeight] = pxColConv[5];

                    pxRowConv[0] = _mm_add_epi16(pxColConv[0], pxRowConv[0]);
                    pxRowConv[1] = _mm_add_epi16(pxColConv[1], pxRowConv[1]);
                    pxRowConv[2] = _mm_add_epi16(pxColConv[2], pxRowConv[2]);
                    pxRowConv[3] = _mm_add_epi16(pxColConv[3], pxRowConv[3]);
                    pxRowConv[4] = _mm_add_epi16(pxColConv[4], pxRowConv[4]);
                    pxRowConv[5] = _mm_add_epi16(pxColConv[5], pxRowConv[5]);
                    srcPtrRowConv += srcDescPtr->strides.hStride;
                }

                // Multiply by convolution factor and write to destination
                pxRowConv[0] = _mm_mulhi_epi16(pxRowConv[0], pxMul); // R0 - R7
                pxRowConv[1] = _mm_mulhi_epi16(pxRowConv[1], pxMul); // R8 - R15
                pxRowConv[2] = _mm_mulhi_epi16(pxRowConv[2], pxMul); // G0 - G7
                pxRowConv[3] = _mm_mulhi_epi16(pxRowConv[3], pxMul); // G8 - G15
                pxRowConv[4] = _mm_mulhi_epi16(pxRowConv[4], pxMul); // B0 - B7
                pxRowConv[5] = _mm_mulhi_epi16(pxRowConv[5], pxMul); // B8 - B15

                __m128i pxRes[4];
                rpp_pln_to_pkd_lower(&pxRowConv[0], shiftMask, &pxRes[0]); // RGB 00-03
                rpp_pln_to_pkd_higher(&pxRowConv[0], shiftMask, &pxRes[1]); // RGB 04-07
                rpp_pln_to_pkd_lower(&pxRowConv[1], shiftMask, &pxRes[2]); // RGB 05-08
                rpp_pln_to_pkd_higher(&pxRowConv[1], shiftMask, &pxRes[3]); // RGB 09-11
                // rpp_pln_to_pkd_new(pxRowConv, pxRes, pxMask1, pxMask2);

                _mm_storeu_si128((__m128i *)dstPtrRow, pxRes[0]);
                _mm_storeu_si128((__m128i *)(dstPtrRow + 12), pxRes[1]);
                _mm_storeu_si128((__m128i *)(dstPtrRow + 24), pxRes[2]);
                _mm_storeu_si128((__m128i *)(dstPtrRow + 36), pxRes[3]);

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;

                // Computation for remaining destination rows
                for(int i = 0; i < roi.xywhROI.roiHeight - 1; i++)
                {
                    Rpp8u *srcPtrRowConv;
                    srcPtrRowConv = srcPtrRow;

                    rpp_reset_variables(pxRowConv);
                    rpp_reset_variables(pxColConv);

                    // Convolution execution
                    // Convolution height loop for pre-computed rows
                    for (int kHeight = 0; kHeight < kernelSizeMinusOne; kHeight++)
                    {
                        pxRowConv[0] = _mm_add_epi16(pxRowConv[0], pxScratchR0[kHeight]);
                        pxRowConv[1] = _mm_add_epi16(pxRowConv[1], pxScratchR1[kHeight]);
                        pxRowConv[2] = _mm_add_epi16(pxRowConv[2], pxScratchG0[kHeight]);
                        pxRowConv[3] = _mm_add_epi16(pxRowConv[3], pxScratchG1[kHeight]);
                        pxRowConv[4] = _mm_add_epi16(pxRowConv[4], pxScratchB0[kHeight]);
                        pxRowConv[5] = _mm_add_epi16(pxRowConv[5], pxScratchB1[kHeight]);
                    }
                    srcPtrRowConv += precomputedRowsIncrement;

                    // Convolution width loop for last row
                    for (int kWidth = 0; kWidth < kernelSize; kWidth++)
                    {
                        // Load 16R, 16G, 16B values
                        pxSrc[0] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth));
                        pxSrc[1] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 12));
                        pxSrc[2] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 24));
                        pxSrc[3] = _mm_loadu_si128((__m128i *)(srcPtrRowConv + 3 * kWidth + 36));

                        rpp_shuffle_and_conv(pxSrc, maskR16, &pxColConv[0]); // R Channel
                        rpp_shuffle_and_conv(pxSrc, maskG16, &pxColConv[2]); // G Channel
                        rpp_shuffle_and_conv(pxSrc, maskB16, &pxColConv[4]); // B Channel
                    }

                    pxScratchR0[kernelSizeMinusOne] = pxColConv[0];
                    pxScratchR1[kernelSizeMinusOne] = pxColConv[1];
                    pxScratchG0[kernelSizeMinusOne] = pxColConv[2];
                    pxScratchG1[kernelSizeMinusOne] = pxColConv[3];
                    pxScratchB0[kernelSizeMinusOne] = pxColConv[4];
                    pxScratchB1[kernelSizeMinusOne] = pxColConv[5];

                    pxRowConv[0] = _mm_add_epi16(pxColConv[0], pxRowConv[0]);
                    pxRowConv[1] = _mm_add_epi16(pxColConv[1], pxRowConv[1]);
                    pxRowConv[2] = _mm_add_epi16(pxColConv[2], pxRowConv[2]);
                    pxRowConv[3] = _mm_add_epi16(pxColConv[3], pxRowConv[3]);
                    pxRowConv[4] = _mm_add_epi16(pxColConv[4], pxRowConv[4]);
                    pxRowConv[5] = _mm_add_epi16(pxColConv[5], pxRowConv[5]);

                    // Multiply by convolution factor and write to destination
                    pxRowConv[0] = _mm_mulhi_epi16(pxRowConv[0], pxMul); // R0 - R7
                    pxRowConv[1] = _mm_mulhi_epi16(pxRowConv[1], pxMul); // R8 - R15
                    pxRowConv[2] = _mm_mulhi_epi16(pxRowConv[2], pxMul); // G0 - G7
                    pxRowConv[3] = _mm_mulhi_epi16(pxRowConv[3], pxMul); // G8 - G15
                    pxRowConv[4] = _mm_mulhi_epi16(pxRowConv[4], pxMul); // B0 - B7
                    pxRowConv[5] = _mm_mulhi_epi16(pxRowConv[5], pxMul); // B8 - B15

                    // Multiply by convolution factor and write to destination
                    __m128i pxRes[4];
                    rpp_pln_to_pkd_lower(&pxRowConv[0], shiftMask, &pxRes[0]); // RGB 00-03
                    rpp_pln_to_pkd_higher(&pxRowConv[0], shiftMask, &pxRes[1]); // RGB 04-07
                    rpp_pln_to_pkd_lower(&pxRowConv[1], shiftMask, &pxRes[2]); // RGB 05-08
                    rpp_pln_to_pkd_higher(&pxRowConv[1], shiftMask, &pxRes[3]); // RGB 09-11
                    // rpp_pln_to_pkd_new(pxRowConv, pxRes, pxMask1, pxMask2);

                    // R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 R4 G4 0
                    _mm_storeu_si128((__m128i *)dstPtrRow, pxRes[0]);
                    _mm_storeu_si128((__m128i *)(dstPtrRow + 12), pxRes[1]);
                    _mm_storeu_si128((__m128i *)(dstPtrRow + 24), pxRes[2]);
                    _mm_storeu_si128((__m128i *)(dstPtrRow + 36), pxRes[3]);

                    for (int kHeight = 0; kHeight < kernelSizeMinusOne; kHeight++)
                    {
                        pxScratchR0[kHeight] = pxScratchR0[kHeight + 1];
                        pxScratchR1[kHeight] = pxScratchR1[kHeight + 1];
                        pxScratchG0[kHeight] = pxScratchG0[kHeight + 1];
                        pxScratchG1[kHeight] = pxScratchG1[kHeight + 1];
                        pxScratchB0[kHeight] = pxScratchB0[kHeight + 1];
                        pxScratchB1[kHeight] = pxScratchB1[kHeight + 1];
                    }

                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }

                srcPtrCol += 48;
                dstPtrCol += 48;
            }
        }
    }

    return RPP_SUCCESS;
}
