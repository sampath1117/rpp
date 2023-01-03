#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_water_src_loc_sse(__m128 &pDstY, __m128 &pDstX, __m128 &pSrcY, __m128 &pSrcX, __m128 &pAmplY, __m128 &pAmplX,
                                      __m128 &pSin, __m128 &pCos, __m128 &pRowLimit, __m128 &pColLimit,
                                      __m128 &pSrcStrideH, Rpp32s *srcLoc, bool hasRGBChannels = false)
{
    pSrcY = _mm_floor_ps(_mm_fmadd_ps(pAmplY, pCos, pDstY));
    pSrcX = _mm_floor_ps(_mm_fmadd_ps(pAmplX, pSin, pDstX));
    pSrcY = _mm_max_ps(_mm_min_ps(pSrcY, pRowLimit), xmm_p0);
    pSrcX = _mm_max_ps(_mm_min_ps(pSrcX, pColLimit), xmm_p0);
    if (hasRGBChannels)
        pSrcX = _mm_mul_ps(pSrcX, xmm_p3);
    __m128i pxSrcLoc = _mm_cvtps_epi32(_mm_fmadd_ps(pSrcY, pSrcStrideH, pSrcX));
    _mm_storeu_si128((__m128i*) srcLoc, pxSrcLoc);
    pDstX = _mm_add_ps(pDstX, xmm_p4);
}

inline void compute_water_src_loc(Rpp32f dstY, Rpp32f dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f amplY, Rpp32f amplX,
                                  Rpp32f sinFactor, Rpp32f cosFactor)
{
    srcY = dstY + amplY * cosFactor;
    srcX = dstX + amplX * sinFactor;
}

template <typename T>
inline void compute_water_interpolation_pkd3_to_pln3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtrTempR, T *dstPtrTempG, T *dstPtrTempB, T *srcPtrChannel, RpptDescPtr srcDescPtr)
{
    srcX = std::min(std::max(roiLTRB->ltrbROI.lt.x, (Rpp32s)srcX), roiLTRB->ltrbROI.rb.x);
    srcY = std::min(std::max(roiLTRB->ltrbROI.lt.y, (Rpp32s)srcY), roiLTRB->ltrbROI.rb.y);

    T *srcPtrTemp;
    srcPtrTemp = srcPtrChannel + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + ((Rpp32s)srcX * srcDescPtr->strides.wStride);
    *dstPtrTempR = *srcPtrTemp++;
    *dstPtrTempG = *srcPtrTemp++;
    *dstPtrTempB = *srcPtrTemp;
}

template <typename T>
inline void compute_water_interpolation_pln3_to_pln3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtr, T *srcPtr, RpptDescPtr srcDescPtr)
{
    srcX = std::min(std::max(roiLTRB->ltrbROI.lt.x, (Rpp32s)srcX), roiLTRB->ltrbROI.rb.x);
    srcY = std::min(std::max(roiLTRB->ltrbROI.lt.y, (Rpp32s)srcY), roiLTRB->ltrbROI.rb.y);

    T *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
    srcPtrTempR = srcPtr + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + (Rpp32s)srcX;
    srcPtrTempG = srcPtrTempR + srcDescPtr->strides.cStride;
    srcPtrTempB = srcPtrTempG + srcDescPtr->strides.cStride;

    dstPtr[0] = *srcPtrTempR;
    dstPtr[1] = *srcPtrTempG;
    dstPtr[2] = *srcPtrTempB;
}

template <typename T>
inline void compute_water_interpolation_pln_to_pln(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtr, T *srcPtr, RpptDescPtr srcDescPtr)
{
    srcX = std::min(std::max(roiLTRB->ltrbROI.lt.x, (Rpp32s)srcX), roiLTRB->ltrbROI.rb.x);
    srcY = std::min(std::max(roiLTRB->ltrbROI.lt.y, (Rpp32s)srcY), roiLTRB->ltrbROI.rb.y);

    T *srcPtrChn = srcPtr;
    T *dstPtrChn = dstPtr;
    for(int i = 0; i < srcDescPtr->c; i++)
    {
        T *srcPtrTemp;
        srcPtrTemp = srcPtrChn + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + (Rpp32s)srcX;
        *dstPtrChn = *srcPtrTemp;
        srcPtrChn += srcDescPtr->strides.cStride;
        dstPtrChn += srcDescPtr->strides.cStride;
    }
}

template <typename T>
inline void compute_water_interpolation_pkd3_to_pkd3(Rpp32f srcY, Rpp32f srcX, RpptROI *roiLTRB, T *dstPtr, T *srcPtr, RpptDescPtr srcDescPtr)
{
    srcX = std::min(std::max(roiLTRB->ltrbROI.lt.x, (Rpp32s)srcX), roiLTRB->ltrbROI.rb.x);
    srcY = std::min(std::max(roiLTRB->ltrbROI.lt.y, (Rpp32s)srcY), roiLTRB->ltrbROI.rb.y);

    T *srcPtrTemp;
    srcPtrTemp = srcPtr + ((Rpp32s)srcY * srcDescPtr->strides.hStride) + (Rpp32s)srcX * srcDescPtr->strides.wStride;

    dstPtr[0] = srcPtrTemp[0];
    dstPtr[1] = srcPtrTemp[1];
    dstPtr[2] = srcPtrTemp[2];
}

RppStatus water_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32f *amplitudeXTensor,
                                  Rpp32f *amplitudeYTensor,
                                  Rpp32f *frequencyXTensor,
                                  Rpp32f *frequencyYTensor,
                                  Rpp32f *phaseXTensor,
                                  Rpp32f *phaseYTensor,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f amplX = amplitudeXTensor[batchCount];
        Rpp32f amplY = amplitudeYTensor[batchCount];
        Rpp32f freqX = frequencyXTensor[batchCount];
        Rpp32f freqY = frequencyYTensor[batchCount];
        Rpp32f phaseX = phaseXTensor[batchCount];
        Rpp32f phaseY = phaseYTensor[batchCount];

        Rpp32u bufferLength = roi.xywhROI.roiWidth;
        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;
        Rpp32u alignedLength = dstDescPtr->w & ~3;   // Align dst width to process 4 dst pixels per iteration
        Rpp32s srcLoc[4] = {0};         // Since 4 dst pixels are processed per iteration
        Rpp32s invalidLoad[4] = {0};    // Since 4 dst pixels are processed per iteration

        __m128 pSrcStrideH = _mm_set1_ps(srcDescPtr->strides.hStride);
        __m128 pRoiLTRB[4];
        pRoiLTRB[0] = _mm_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm_set1_ps(roiLTRB.ltrbROI.rb.y);

        __m128 pAmplX = _mm_set1_ps(amplX);
        __m128 pAmplY = _mm_set1_ps(amplY);
        __m128 pFreqX = _mm_set1_ps(freqX);
        __m128 pFreqY = _mm_set1_ps(freqY);
        __m128 pPhaseX = _mm_set1_ps(phaseX);
        __m128 pPhaseY = _mm_set1_ps(phaseY);

        // water with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcPtrRow, *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            srcPtrRow = srcPtrChannel;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f dstY = i;
                __m128 pDstX = xmm_pDstLocInit;
                __m128 pDstY = _mm_set1_ps(dstY);
                Rpp32f sinFactor = std::sin((freqX * dstY) + phaseX);
                __m128 pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_fmadd_ps(pFreqY, pDstX, pPhaseY), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pAmplY, pAmplX, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLoc, true);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, srcLoc, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);

                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f dstX = vectorLoopCount;
                    Rpp32f cosFactor = std::cos((freqY * dstX) + phaseY);
                    Rpp32f srcX, srcY;
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor);
                    compute_water_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // water with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrChannelR, *srcPtrChannelG, *srcPtrChannelB;
            srcPtrChannelR = srcPtrChannel;
            srcPtrChannelG = srcPtrChannelR + srcDescPtr->strides.cStride;
            srcPtrChannelB = srcPtrChannelG + srcDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstY = i;
                __m128 pDstX = xmm_pDstLocInit;
                __m128 pDstY = _mm_set1_ps(dstY);
                Rpp32f sinFactor = std::sin((freqX * dstY) + phaseX);
                __m128 pSinFactor = _mm_set1_ps(sinFactor);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow[3];
                    sincos_ps(_mm_fmadd_ps(pFreqY, pDstX, pPhaseY), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pAmplY, pAmplX, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLoc);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrChannelR, srcLoc, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrChannelG, srcLoc, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrChannelB, srcLoc, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f dstX = vectorLoopCount;
                    Rpp32f cosFactor = std::cos((freqY * dstX) + phaseY);
                    Rpp32f srcX, srcY;
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor);
                    compute_water_interpolation_pln3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannelR, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f dstY = i;
                __m128 pDstX = xmm_pDstLocInit;
                __m128 pDstY = _mm_set1_ps(dstY);
                Rpp32f sinFactor = std::sin((freqX * dstY) + phaseX);
                __m128 pSinFactor = _mm_set1_ps(sinFactor);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_fmadd_ps(pFreqY, pDstX, pPhaseY), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pAmplY, pAmplX, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLoc, true);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrChannel, srcLoc, pRow);
                    rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f dstX = vectorLoopCount;
                    Rpp32f cosFactor = std::cos((freqY * dstX) + phaseY);
                    Rpp32f srcX, srcY;
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor);
                    compute_water_interpolation_pkd3_to_pkd3(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Water with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                Rpp32f dstY = i;
                __m128 pDstX = xmm_pDstLocInit;
                __m128 pDstY = _mm_set1_ps(dstY);
                Rpp32f sinFactor = std::sin((freqX * dstY) + phaseX);
                __m128 pSinFactor = _mm_set1_ps(sinFactor);

                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    sincos_ps(_mm_fmadd_ps(pFreqY, pDstX, pPhaseY), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pAmplY, pAmplX, pSinFactor, pCosFactor, pRoiLTRB[3], pRoiLTRB[2], pSrcStrideH, srcLoc);
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrChannel;
                    dstPtrTempChn = dstPtrTemp;
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempChn, srcLoc, pRow);
                        rpp_simd_store(rpp_store4_u8_to_u8, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f dstX = vectorLoopCount;
                    Rpp32f cosFactor = std::cos((freqY * dstX) + phaseY);
                    Rpp32f srcX, srcY;
                    compute_water_src_loc(dstY, dstX, srcY, srcX, amplY, amplX, sinFactor, cosFactor);
                    compute_water_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}