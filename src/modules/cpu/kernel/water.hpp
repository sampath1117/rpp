#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_water_src_loc_sse(__m128 &pDstY, __m128 &pDstX, __m128 &pSrcY, __m128 &pSrcX, __m128 &pAmplY, __m128 &pAmplX,
                                      __m128 &pSin, __m128 &pCos, __m128 &pPhaseY, __m128 &pPhaseX)
{
    pSrcY = _mm_add_ps(pDstY, _mm_fmadd_ps(pAmplY, pCos, pPhaseY));
    pSrcX = _mm_add_ps(pDstX, _mm_fmadd_ps(pAmplX, pSin, pPhaseX));
}

inline void compute_water_src_loc(Rpp32f dstY, Rpp32f dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f amplY, Rpp32f amplX,
                                  Rpp32f sinFactor, Rpp32f cosFactor, Rpp32f phaseY, Rpp32f phaseX)
{
    srcY = dstY + (amplY * cosFactor) + phaseY;
    srcX = dstX + (amplX * sinFactor) + phaseX;
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

        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
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

                Rpp32f sinFactor = std::sin(freqX * i);
                __m128 pSinFactor = _mm_set1_ps(sinFactor);
                __m128 pDstX = xmm_pDstLocInit;
                __m128 pDstY = _mm_set1_ps(i);
                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pCosFactor, pDummy, pSrcX, pSrcY;
                    __m128i pRow;
                    sincos_ps(_mm_mul_ps(pFreqY, pDstX), &pDummy, &pCosFactor);
                    compute_water_src_loc_sse(pDstY, pDstX, pSrcY, pSrcX, pAmplY, pAmplX, pSinFactor, pCosFactor, pPhaseY, pPhaseX);
                    compute_generic_nn_srclocs_and_validate_sse(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLoc, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3, srcPtrChannel, srcLoc, invalidLoad, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);

                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                    pDstX = _mm_add_ps(pDstX, xmm_p4);
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f cosFactor = std::cos(freqY * vectorLoopCount);
                    Rpp32f srcX, srcY;
                    compute_water_src_loc(i, vectorLoopCount, srcY, srcX, amplY, amplX, sinFactor, cosFactor, phaseX, phaseY);
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
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

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 p[6];
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp += 3;
                }

                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}