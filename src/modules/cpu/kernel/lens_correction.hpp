#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

/************* lens_correction helpers *************/

inline void compute_lens_correction_src_loc(Rpp32s dstRowLoc, Rpp32f dstRowLocSquared, Rpp32s dstColLoc, Rpp32f &srcRowLoc, Rpp32f &srcColLoc, Rpp32f &zoom, Rpp32f &invCorrectionRadius, Rpp32s roiHalfHeight, Rpp32s roiHalfWidth)
{
    dstColLoc -= roiHalfWidth;
    float distance = (float)(sqrt(dstColLoc * dstColLoc + dstRowLocSquared)) * invCorrectionRadius;

    float theta;
    if (distance == 0)
        theta = 1.0;
    else
        theta = atan(distance) / distance;

    srcColLoc = (roiHalfWidth + theta * dstColLoc * zoom);
    srcRowLoc = (roiHalfHeight + theta * dstRowLoc * zoom);
}

inline void compute_lens_correction_src_loc_avx(__m256 &pDstRowLoc, __m256 &pDstRowLocSquared, __m256 &pDstColLoc, __m256 &pSrcRowLoc, __m256 &pSrcColLoc, __m256 &pZoom, __m256 &pinvCorrectionRadius, __m256 &pHalfHeight, __m256 &pHalfWidth)
{
    __m256 pDstColLocNew = _mm256_sub_ps(pDstColLoc, pHalfWidth);
    pDstColLoc = _mm256_add_ps(pDstColLoc, avx_p8);
    __m256 pDistance = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(pDstColLocNew, pDstColLocNew), pDstRowLocSquared));
    pDistance = _mm256_mul_ps(pDistance, pinvCorrectionRadius);
    __m256 pMask =  _mm256_cmp_ps(pDistance, avx_p0, _CMP_EQ_OQ);
    __m256 pTheta = _mm256_blendv_ps(_mm256_div_ps(atan_ps_avx(pDistance), pDistance), avx_p1, pMask);
    __m256 pMulFactor = _mm256_mul_ps(pTheta, pZoom);
    pSrcColLoc = _mm256_add_ps(pHalfWidth, _mm256_mul_ps(pDstColLocNew, pMulFactor));
    pSrcRowLoc = _mm256_add_ps(pHalfHeight, _mm256_mul_ps(pDstRowLoc, pMulFactor));
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
        Rpp32f invCorrectionRadius = strength / sqrt(width * width + height * height) ;

        __m256 pZoom, pinvCorrectionRadius, pHalfWidth, pHalfHeight;
        pZoom = _mm256_set1_ps(zoom);
        pinvCorrectionRadius = _mm256_set1_ps(invCorrectionRadius);
        pHalfWidth = _mm256_set1_ps(roiHalfWidth);
        pHalfHeight = _mm256_set1_ps(roiHalfHeight);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
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

        // Lens Correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }

                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *dstPtrTemp = dstPtrRow;
                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_1c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8u>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_u8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus lens_correction_bilinear_f32_f32_host_tensor(Rpp32f *srcPtr,
                                                       RpptDescPtr srcDescPtr,
                                                       Rpp32f *dstPtr,
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
        Rpp32f invCorrectionRadius = strength / sqrt(width * width + height * height) ;

        __m256 pZoom, pinvCorrectionRadius, pHalfWidth, pHalfHeight;
        pZoom = _mm256_set1_ps(zoom);
        pinvCorrectionRadius = _mm256_set1_ps(invCorrectionRadius);
        pHalfWidth = _mm256_set1_ps(roiHalfWidth);
        pHalfHeight = _mm256_set1_ps(roiHalfHeight);

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
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

        // Lens Correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }

                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp32f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f32pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp32f *dstPtrTemp = dstPtrRow;
                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_1c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp32f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f32pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus lens_correction_bilinear_f16_f16_host_tensor(Rpp16f *srcPtr,
                                                       RpptDescPtr srcDescPtr,
                                                       Rpp16f *dstPtr,
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
        Rpp32f invCorrectionRadius = strength / sqrt(width * width + height * height) ;

        __m256 pZoom, pinvCorrectionRadius, pHalfWidth, pHalfHeight;
        pZoom = _mm256_set1_ps(zoom);
        pinvCorrectionRadius = _mm256_set1_ps(invCorrectionRadius);
        pHalfWidth = _mm256_set1_ps(roiHalfWidth);
        pHalfHeight = _mm256_set1_ps(roiHalfHeight);

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
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

        // Lens Correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }

                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp16f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp16f *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store24_f32pln3_to_f16pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp16f *dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp16f *dstPtrTemp = dstPtrRow;
                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_1c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp16f>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    rpp_simd_store(rpp_store8_f32pln1_to_f16pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus lens_correction_bilinear_i8_i8_host_tensor(Rpp8s *srcPtr,
                                                     RpptDescPtr srcDescPtr,
                                                     Rpp8s *dstPtr,
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
        Rpp32f invCorrectionRadius = strength / sqrt(width * width + height * height) ;

        __m256 pZoom, pinvCorrectionRadius, pHalfWidth, pHalfHeight;
        pZoom = _mm256_set1_ps(zoom);
        pinvCorrectionRadius = _mm256_set1_ps(invCorrectionRadius);
        pHalfWidth = _mm256_set1_ps(roiHalfWidth);
        pHalfHeight = _mm256_set1_ps(roiHalfHeight);

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage;
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u bufferLength = roi.xywhROI.roiWidth;
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

        // Lens Correction with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pkd3_to_pln3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }

                for (; vectorLoopCount < dstDescPtr->w; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < height; i++)
            {
                Rpp8s *dstPtrTemp;
                dstPtrTemp = dstPtrRow;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, true);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pkd3_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln3pkd3_to_pkd3(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp, srcPtrChannel, srcDescPtr);
                    dstPtrTemp += 3;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < height; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;

                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[12], pDst[3];
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_3c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW, srcDescPtr->c, false);
                    rpp_simd_load(rpp_generic_bilinear_load_3c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_3c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_3c_avx(pDst);
                    rpp_simd_store(rpp_store24_f32pln3_to_i8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pDst); // Store dst pixels
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTempR++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Lens Correction without fused output-layout toggle single channel (NCHW -> NCHW)
        else if ((srcDescPtr->c == 1) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8s *dstPtrTemp = dstPtrRow;
                Rpp32f srcRowLoc, srcColLoc, dstRowLoc, dstRowLocSquared;
                __m256 pSrcRowLoc, pSrcColLoc, pDstRowLoc, pDstRowLocSquared, pDstColLoc;
                dstRowLoc = (i - roiHalfHeight);
                dstRowLocSquared = dstRowLoc * dstRowLoc;

                pDstRowLoc = _mm256_set1_ps(dstRowLoc);
                pDstRowLocSquared = _mm256_set1_ps(dstRowLocSquared);
                pDstColLoc = avx_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrc[4], pDst;
                    compute_lens_correction_src_loc_avx(pDstRowLoc, pDstRowLocSquared, pDstColLoc, pSrcRowLoc, pSrcColLoc, pZoom, pinvCorrectionRadius, pHalfHeight, pHalfWidth);
                    compute_generic_bilinear_srclocs_1c_avx(pSrcRowLoc, pSrcColLoc, srcLocs, pBilinearCoeffs, pSrcStrideH, pxSrcStridesCHW);
                    rpp_simd_load(rpp_generic_bilinear_load_1c_avx<Rpp8s>, srcPtrChannel, srcDescPtr, srcLocs, pSrcRowLoc, pSrcColLoc, pRoiLTRB, pSrc);  // Load input pixels required for bilinear interpolation
                    compute_bilinear_interpolation_1c_avx(pSrc, pBilinearCoeffs, pDst); // Compute Bilinear interpolation
                    compute_offset_i8_1c_avx(pDst);
                    rpp_simd_store(rpp_store8_f32pln1_to_i8pln1_avx, dstPtrTemp, pDst); // Store dst pixels
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    compute_lens_correction_src_loc(dstRowLoc, dstRowLocSquared, vectorLoopCount, srcRowLoc, srcColLoc, zoom, invCorrectionRadius, roiHalfHeight, roiHalfWidth);
                    compute_generic_bilinear_interpolation_pln_to_pln(srcRowLoc, srcColLoc, &roiLTRB, dstPtrTemp++, srcPtrChannel, srcDescPtr, dstDescPtr);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}