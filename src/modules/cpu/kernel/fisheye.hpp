/*
MIT License

Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline void compute_fisheye_src_loc_avx(__m256 &pDstY, __m256 &pDstX, __m256 &pSrcY, __m256 &pSrcX, __m256 &pHeight, __m256 &pWidth)
{
    __m256 pNormX, pNormY, pDist;
    pNormX = _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(avx_p2, pDstX), pWidth), avx_p1);        //  (static_cast<Rpp32f>((2.0 * dstX)) / width) - 1;
    pNormY = _mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(avx_p2, pDstY), pHeight), avx_p1);       //  (static_cast<Rpp32f>((2.0 * dstY)) / height) - 1;
    pDist = _mm256_sqrt_ps(_mm256_fmadd_ps(pNormX, pNormX, _mm256_mul_ps(pNormY, pNormY)));     //  std::sqrt((normX * normX) + (normY * normY));
    
    __m256 pDistNew, pTheta, pSinFactor, pCosFactor;
    pDistNew = _mm256_sqrt_ps(_mm256_sub_ps(avx_p1, _mm256_mul_ps(pDist, pDist)));              //  std::sqrt(1.0 - dist * dist);
    pDistNew = _mm256_mul_ps(_mm256_add_ps(pDist, _mm256_sub_ps(avx_p1, pDistNew)), avx_p1op2); //  (dist + (1.0 - distNew)) * 0.5f; 
    pTheta = atan2_ps(pNormY, pNormX);                                                          //  std::atan2(normY, normX);
    sincos_ps(pTheta, &pSinFactor, &pCosFactor);

    pSrcX = _mm256_mul_ps(_mm256_mul_ps(_mm256_fmadd_ps(pDistNew, pCosFactor, avx_p1), pWidth), avx_p1op2);
    pSrcY = _mm256_mul_ps(_mm256_mul_ps(_mm256_fmadd_ps(pDistNew, pSinFactor, avx_p1), pHeight), avx_p1op2);
    pSrcX = _mm256_blendv_ps(avx_pMinus1, pSrcX, _mm256_cmp_ps(pSrcX, pSrcX, _CMP_ORD_Q));
    pSrcY = _mm256_blendv_ps(avx_pMinus1, pSrcY, _mm256_cmp_ps(pSrcY, pSrcY, _CMP_ORD_Q));

    __m256 pMask1, pMask2;
    pMask1 = _mm256_and_ps(_mm256_cmp_ps(pDist, avx_p0, _CMP_GE_OQ), _mm256_cmp_ps(pDist, avx_p1, _CMP_LE_OQ));
    pMask2 = _mm256_and_ps(pMask1, _mm256_cmp_ps(pDistNew, avx_p1, _CMP_LE_OQ));
    pSrcX = _mm256_blendv_ps(avx_pMinus1, pSrcX, pMask2);
    pSrcY = _mm256_blendv_ps(avx_pMinus1, pSrcY, pMask2);
    pDstX = _mm256_add_ps(pDstX, avx_p8);
}

inline void compute_fisheye_src_loc(Rpp32f dstY, Rpp32f dstX, Rpp32f &srcY, Rpp32f &srcX, Rpp32s &height, Rpp32s &width)
{
    Rpp32f normX = (static_cast<Rpp32f>((2.0 * dstX)) / width) - 1;
    Rpp32f normY = (static_cast<Rpp32f>((2.0 * dstY)) / height) - 1;
    Rpp32f dist = std::sqrt((normX * normX) + (normY * normY));
    srcX = -1;
    srcY = -1;
    if ((dist >= 0.0) && (dist <= 1.0))
    {
        Rpp32f distNew = std::sqrt(1.0 - dist * dist);
        distNew = (dist + (1.0 - distNew)) * 0.5f;
        if (distNew <= 1.0)
        {
            Rpp32f theta = std::atan2(normY, normX);
            Rpp32f newX = distNew * std::cos(theta);
            Rpp32f newY = distNew * std::sin(theta);
            srcX = (((newX + 1) * width) * 0.5f);
            srcY = (((newY + 1) * height) * 0.5f);
        }
    }
}

RppStatus fisheye_u8_u8_host_tensor(Rpp8u *srcPtr,
                                    RpptDescPtr srcDescPtr,
                                    Rpp8u *dstPtr,
                                    RpptDescPtr dstDescPtr,
                                    RpptROIPtr roiTensorPtrSrc,
                                    RpptRoiType roiType,
                                    RppLayoutParams layoutParams,
                                    rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi, roiLTRB;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);
        compute_ltrb_from_xywh_host(&roi, &roiLTRB);
        Rpp32u bufferLength = roi.xywhROI.roiWidth;

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32s vectorIncrementPerChannel = 8;
        Rpp32s vectorIncrementPkd = 24;
        Rpp32u alignedLength = bufferLength & ~7;   // Align dst width to process 8 dst pixels per iteration
        Rpp32s srcLocArray[8] = {0};                // Since 8 dst pixels are processed per iteration
        Rpp32s invalidLoad[8] = {0};                // Since 8 dst pixels are processed per iteration

#if __AVX2__
        __m256 pSrcStrideH = _mm256_set1_ps(srcDescPtr->strides.hStride);
        __m256 pRoiLTRB[4], pWidth, pHeight;
        pRoiLTRB[0] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.x);
        pRoiLTRB[1] = _mm256_set1_ps(roiLTRB.ltrbROI.lt.y);
        pRoiLTRB[2] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.x);
        pRoiLTRB[3] = _mm256_set1_ps(roiLTRB.ltrbROI.rb.y);
        pWidth = _mm256_set1_ps(roi.xywhROI.roiWidth);
        pHeight = _mm256_set1_ps(roi.xywhROI.roiHeight);
#endif

        // fisheye with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
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

                Rpp32f dstX, dstY;
                dstY = static_cast<Rpp32f>(i);
                Rpp32s vectorLoopCount = 0;
#if __AVX2__
                __m256 pDstX, pDstY;
                pDstX = avx_pDstLocInit;
                pDstY = _mm256_set1_ps(dstY);
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m256 pSrcX, pSrcY;
                    __m256i pRow;
                    compute_fisheye_src_loc_avx(pDstY, pDstX, pSrcY, pSrcX, pHeight, pWidth);
                    compute_generic_nn_srclocs_and_validate_avx(pSrcY, pSrcX, pRoiLTRB, pSrcStrideH, srcLocArray, invalidLoad, true);
                    rpp_simd_load(rpp_generic_nn_load_u8pkd3_avx, srcPtrChannel, srcLocArray, invalidLoad, pRow);
                    rpp_simd_store(rpp_store24_u8pkd3_to_u8pln3_avx, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    Rpp32f srcX, srcY, cosFactor;
                    dstX = static_cast<Rpp32f>(vectorLoopCount);
                    compute_fisheye_src_loc(dstY, dstX, srcY, srcX, roi.xywhROI.roiHeight, roi.xywhROI.roiWidth);
                    compute_generic_nn_interpolation_pkd3_to_pln3(srcY, srcX, &roiLTRB, dstPtrTempR++, dstPtrTempG++, dstPtrTempB++, srcPtrChannel, srcDescPtr);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // fisheye with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32u alignedLength = (bufferLength / 48) * 48;
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB, *dstPtrRow;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            dstPtrRow = dstPtrChannel;
            for(int i = 0; i < roi.xywhROI.roiHeight; i++)
            {
                Rpp8u *srcPtrTempR, *srcPtrTempG, *srcPtrTempB, *dstPtrTemp;
                srcPtrTempR = srcPtrRowR;
                srcPtrTempG = srcPtrRowG;
                srcPtrTempB = srcPtrRowB;
                dstPtrTemp = dstPtrRow;
                int vectorLoopCount = 0;
#if __AVX2__
                for (; vectorLoopCount < alignedLength; vectorLoopCount+=16)
                {
                    __m128i px[3];
                    rpp_simd_load(rpp_load48_u8pln3_to_u8pln3, srcPtrTempR, srcPtrTempG, srcPtrTempB, px);    // simd loads
                    rpp_simd_store(rpp_store48_u8pln3_to_u8pkd3, dstPtrTemp, px);    // simd stores
                    srcPtrTempR += 16;
                    srcPtrTempG += 16;
                    srcPtrTempB += 16;
                    dstPtrTemp += 48;
                }
#endif
                for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                {
                    dstPtrTemp[0] = *srcPtrTempR;
                    dstPtrTemp[1] = *srcPtrTempG;
                    dstPtrTemp[2] = *srcPtrTempB;
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
    }

    return RPP_SUCCESS;
}