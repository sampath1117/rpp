#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

/************* lens_correction helpers *************/

inline void compute_lens_correction_src_loc(Rpp32s dstY, Rpp32f dstYSquared, Rpp32s dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32f &zoom, Rpp32f &correctionRadius, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    dstX -= roiHalfWidth;
    float r = (float)(sqrt(dstX * dstX + dstYSquared)) / correctionRadius;

    float theta;
    if (r == 0)
        theta = 1.0;
    else
        theta = atan(r) / r;

    srcX = (roiHalfWidth + theta * dstX * zoom);
    srcY = (roiHalfHeight + theta * dstY * zoom);
}

inline void compute_lens_correction_src_loc_avx(__m256 &pDstY, __m256 &pDstYSquared, __m256 &pDstX, __m256 &pSrcY, __m256 &pSrcX, __m256 &pZoom, __m256 &pCorrectionRadius, __m256 &pHalfHeight, __m256 &pHalfWidth)
{
    __m256 pColLoc = _mm256_sub_ps(pDstX, pHalfWidth);
    pDstX = _mm256_add_ps(pDstX, avx_pDstLocInit);

    __m256 pDistance = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(pColLoc, pColLoc), pDstYSquared));
    pDistance = _mm256_mul_ps(pDistance, pCorrectionRadius);
    __m256 pMask =  _mm256_cmp_ps(pDistance, avx_p0, _CMP_EQ_OQ);
    __m256 pTheta = _mm256_blendv_ps(_mm256_div_ps(atan_ps_avx(pDistance), pDistance), avx_p1, pMask);
    __m256 pFactor = _mm256_mul_ps(pTheta, pZoom);

    pSrcX = _mm256_add_ps(pHalfWidth, _mm256_mul_ps(pFactor, pColLoc));
    pSrcY = _mm256_add_ps(pHalfHeight, _mm256_mul_ps(pFactor, pDstY));
}

// /************* BILINEAR INTERPOLATION *************/

RppStatus lens_correction_bilinear_u8_u8_host_tensor(Rpp8u *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp8u *dstPtr,
                                                     RpptDescPtr dstDescPtr,
                                                     Rpp32f *strengthTensor,
                                                     Rpp32f *zoomTensor,
                                                     RpptROIPtr roiTensorPtrSrc,
                                                     RpptRoiType roiType,
                                                     RppLayoutParams srcLayoutParams)
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
        Rpp32s roiHalfWidth = roi.xywhROI.roiWidth >> 1;
        Rpp32s roiHalfHeight = roi.xywhROI.roiHeight >> 1;

        Rpp32f zoom = zoomTensor[batchCount];
        Rpp32f strength = strengthTensor[batchCount];
        if (strength == 0.0f)
            strength = 0.000001;
        Rpp32s height = roi.xywhROI.roiHeight;
        Rpp32s width = roi.xywhROI.roiWidth;
        Rpp32f correctionRadius = sqrt(width * width + height * height) / strength;

        __m256 pZoom, pCorrectionRadius, pHalfWidth, pHalfHeight;
        pZoom = _mm256_set1_ps(zoom);
        pCorrectionRadius = _mm256_set1_ps(correctionRadius);
        pHalfWidth = _mm256_set1_ps(roiHalfWidth);
        pHalfHeight = _mm256_set1_ps(roiHalfHeight);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * srcLayoutParams.bufferMultiplier;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 8 dst pixels per iteration

        __m256 pBilinearCoeffs[4];
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pRoiLTRB[4];
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        __m256i pxSrcStridesCHW[3];
        pxSrcStridesCHW[0] = _mm256_set1_epi32(srcDescPtr->strides.cStride);
        pxSrcStridesCHW[1] = _mm256_set1_epi32(srcDescPtr->strides.hStride);
        pxSrcStridesCHW[2] = _mm256_set1_epi32(srcDescPtr->strides.wStride);
        RpptBilinearNbhoodLocsVecLen8 srcLocs;

        // Lens Correction with fused output-layout toggle (NCHW -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcX, srcY;
                __m256 pSrcX, pSrcY, pDstX, pDstY, pDstYSquared;

                Rpp32f dstY, dstYSquared;
                dstY = (i - roiHalfHeight);
                dstYSquared = dstY * dstY;

                pDstY = _mm256_set1_ps(dstY);
                pDstYSquared = _mm256_set1_ps(dstYSquared);
                pDstX = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstY, pDstYSquared, pDstX, pSrcY, pSrcX, pZoom, pCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcY, pSrcX, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcY, pSrcX, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstY, dstYSquared, vectorLoopCount, srcY, srcX, zoom, correctionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcY, srcX, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

