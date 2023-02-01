#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus normalize_u8_f32_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp32f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *offsetTensor,
                                       Rpp32f *multiplierTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];

        // For PKD3-PKD3 case
        // set mean as multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]] 0 multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]]
        // set invStdDev as offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]] 1 offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]]
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            pCMNParams[0] = _mm256_setr_ps(multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f, multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f);
            pCMNParams[1] = _mm256_setr_ps(offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f, offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f);
        }
        else
        {
            for(int pos = 0; pos < numRegs; pos += 2)
            {
                pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
                pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
                cmnParamLoc++;
            }
        }

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        Rpp32f *dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow;
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                    compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR = ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                    *dstPtrTempG = ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                    *dstPtrTempB = ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp32f *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp32f *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = ((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                    dstPtrTemp[1] = ((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                    dstPtrTemp[2] = ((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow;
            Rpp32f *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                Rpp32f *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[8];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pkd3_avx, srcPtrTemp, p);    // simd loads
                    compute_cmn_48_rgb_host(p, pCMNParams);    // cmn adjustment
                    rpp_simd_store(rpp_store48_f32pkd3_to_f32pkd3_avx, dstPtrTemp, p);    // simd stores

                    dstPtrTemp += vectorIncrement;
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] = ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                    dstPtrTemp[1] = ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                    dstPtrTemp[2] = ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                Rpp32f *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp32f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                        rpp_simd_store(rpp_store16_f32_to_f32_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp = ((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]];

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus normalize_u8_f16_host_tensor(Rpp8u *srcPtr,
                                       RpptDescPtr srcDescPtr,
                                       Rpp16f *dstPtr,
                                       RpptDescPtr dstDescPtr,
                                       Rpp32f *offsetTensor,
                                       Rpp32f *multiplierTensor,
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

        Rpp32u cmnParamLoc = srcDescPtr->c * batchCount;
        Rpp32u cmnParamLocs[3] = {cmnParamLoc, cmnParamLoc + 1, cmnParamLoc + 2};
        Rpp32s numRegs = 2 * srcDescPtr->c;
        __m256 pCMNParams[numRegs];

        // For PKD3-PKD3 case
        // set mean as multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]] 0 multiplierTensor[cmnParamLocs[0]] multiplierTensor[cmnParamLocs[1]] multiplierTensor[cmnParamLocs[2]]
        // set invStdDev as offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]] 1 offsetTensor[cmnParamLocs[0]] offsetTensor[cmnParamLocs[1]] offsetTensor[cmnParamLocs[2]]
        if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            pCMNParams[0] = _mm256_setr_ps(multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f, multiplierTensor[cmnParamLocs[0]], multiplierTensor[cmnParamLocs[1]], multiplierTensor[cmnParamLocs[2]], 0.0f);
            pCMNParams[1] = _mm256_setr_ps(offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f, offsetTensor[cmnParamLocs[0]], offsetTensor[cmnParamLocs[1]], offsetTensor[cmnParamLocs[2]], 1.0f);
        }
        else
        {
            for(int pos = 0; pos < numRegs; pos += 2)
            {
                pCMNParams[pos] = _mm256_set1_ps(multiplierTensor[cmnParamLoc]);
                pCMNParams[pos + 1] = _mm256_set1_ps(offsetTensor[cmnParamLoc]);
                cmnParamLoc++;
            }
        }

        Rpp8u *srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp16f *dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp8u *srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        Rpp16f *dstPtrChannel = dstPtrImage;

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u alignedLength = (bufferLength / 48) * 48;
        Rpp32u vectorIncrement = 48;
        Rpp32u vectorIncrementPerChannel = 16;

        // Normalize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow;
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                srcPtrTemp = srcPtrRow;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pln3_avx, srcPtrTemp, p);    //simd loads
                    compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, p);    // simd stores

                    srcPtrTemp += vectorIncrement;
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    *dstPtrTempR =  (Rpp16f) ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                    *dstPtrTempG =  (Rpp16f) ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                    *dstPtrTempB =  (Rpp16f) ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                    srcPtrTemp += 3;
                    dstPtrTempR++;
                    dstPtrTempG++;
                    dstPtrTempB++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Normalize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            Rpp16f *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                Rpp16f *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    rpp_simd_load(rpp_load48_u8pln3_to_f32pln3_avx, srcPtrTempR, srcPtrTempG, srcPtrTempB, p);    // simd loads
                    compute_cmn_48_host(p, pCMNParams);    // cmn adjustment
                    rpp_simd_store(rpp_store48_f32pln3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                    srcPtrTempR += vectorIncrementPerChannel;
                    srcPtrTempG += vectorIncrementPerChannel;
                    srcPtrTempB += vectorIncrementPerChannel;
                    dstPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] =  (Rpp16f) ((Rpp32f)*srcPtrTempR * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                    dstPtrTemp[1] =  (Rpp16f) ((Rpp32f)*srcPtrTempG * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                    dstPtrTemp[2] =  (Rpp16f) ((Rpp32f)*srcPtrTempB * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];

                    srcPtrTempR++;
                    srcPtrTempG++;
                    srcPtrTempB++;
                    dstPtrTemp += 3;
                }
                srcPtrRowR += srcDescPtr->strides.hStride;
                srcPtrRowG += srcDescPtr->strides.hStride;
                srcPtrRowB += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Normalize without fused output-layout toggle (NHWC -> NHWC)
        else if((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcPtrRow;
            Rpp16f *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTemp;
                Rpp16f *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrement)
                {
                    __m256 p[8];
                    rpp_simd_load(rpp_load48_u8pkd3_to_f32pkd3_avx, srcPtrTemp, p);    // simd loads
                    compute_cmn_48_rgb_host(p, pCMNParams);    // cmn adjustment
                    rpp_simd_store(rpp_store48_f32pkd3_to_f16pkd3_avx, dstPtrTemp, p);    // simd stores

                    dstPtrTemp += vectorIncrement;
                    srcPtrTemp += vectorIncrement;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount += 3)
                {
                    dstPtrTemp[0] =  (Rpp16f) ((Rpp32f)srcPtrTemp[0] * multiplierTensor[cmnParamLocs[0]]) + offsetTensor[cmnParamLocs[0]];
                    dstPtrTemp[1] =  (Rpp16f) ((Rpp32f)srcPtrTemp[1] * multiplierTensor[cmnParamLocs[1]]) + offsetTensor[cmnParamLocs[1]];
                    dstPtrTemp[2] =  (Rpp16f) ((Rpp32f)srcPtrTemp[2] * multiplierTensor[cmnParamLocs[2]]) + offsetTensor[cmnParamLocs[2]];
                    srcPtrTemp += 3;
                    dstPtrTemp += 3;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Normalize without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32u alignedLength = (bufferLength / 16) * 16;
            for(int c = 0; c < layoutParams.channelParam; c++)
            {
                Rpp8u *srcPtrRow;
                Rpp16f *dstPtrRow;
                srcPtrRow = srcPtrChannel;
                dstPtrRow = dstPtrChannel;

                for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                {
                    Rpp8u *srcPtrTemp;
                    Rpp16f *dstPtrTemp;
                    srcPtrTemp = srcPtrRow;
                    dstPtrTemp = dstPtrRow;

                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                    {
                        __m256 p[2];
                        rpp_simd_load(rpp_load16_u8_to_f32_avx, srcPtrTemp, p);    // simd loads
                        compute_cmn_16_host(p, &pCMNParams[2 * c]);    // cmn adjustment
                        rpp_simd_store(rpp_store16_f32_to_f16_avx, dstPtrTemp, p);    // simd stores

                        srcPtrTemp += vectorIncrementPerChannel;
                        dstPtrTemp += vectorIncrementPerChannel;
                    }
                    for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                    {
                        *dstPtrTemp =  (Rpp16f) ((Rpp32f)*srcPtrTemp * multiplierTensor[cmnParamLocs[c]]) + offsetTensor[cmnParamLocs[c]];

                        srcPtrTemp++;
                        dstPtrTemp++;
                    }
                    srcPtrRow += srcDescPtr->strides.hStride;
                    dstPtrRow += dstDescPtr->strides.hStride;
                }
                srcPtrChannel += srcDescPtr->strides.cStride;
                dstPtrChannel += dstDescPtr->strides.cStride;
            }
        }
    }

    return RPP_SUCCESS;
}