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
#include "rpp_cpu_filter.hpp"

inline void sobel_filter_generic_tensor(Rpp8u **srcPtrTemp, Rpp8u *dstPtrTemp, Rpp32s columnIndex,
                                        Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                        Rpp32f *filterTensor, Rpp32u channels = 1)
{
    Rpp32f accum = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    for (int i = 0; i < rowKernelLoopLimit; i++)
        for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
            accum += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterTensor[i * kernelSize + j];

    saturate_pixel(std::nearbyintf(accum), dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them separately
inline void process_left_border_columns_pln_pln(Rpp8u **srcPtrTemp, Rpp8u *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterTensor)
{
    for (int k = 0; k < padLength; k++)
    {
        sobel_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterTensor, 1);
        dstPtrTemp++;
    }
}

inline void sobel_filter_generic_tensor(Rpp8u **srcPtrTemp, Rpp8u *dstPtrTemp, Rpp32s columnIndex,
                                        Rpp32u kernelSize, Rpp32u padLength, Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit,
                                        Rpp32f *filterXTensor, Rpp32f *filterYTensor, Rpp32u channels = 1)
{
    Rpp32f accumX = 0.0f;
    Rpp32f accumY = 0.0f;
    Rpp32s columnKernelLoopLimit = kernelSize;

    // find the colKernelLoopLimit based on columnIndex
    get_kernel_loop_limit(columnIndex, columnKernelLoopLimit, padLength, unpaddedWidth);
    for (int i = 0; i < rowKernelLoopLimit; i++)
    {
        for (int j = 0, k = 0 ; j < columnKernelLoopLimit; j++, k += channels)
        {
            accumX += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterXTensor[i * kernelSize + j];
            accumY += static_cast<Rpp32f>(srcPtrTemp[i][k]) * filterYTensor[i * kernelSize + j];
        }
    }

    Rpp32f accum = sqrt((accumX * accumX) + (accumY * accumY));
    saturate_pixel(std::nearbyintf(accum), dstPtrTemp);
}

// process padLength number of columns in each row
// left border pixels in image which does not have required pixels in 3x3 kernel, process them separately
inline void process_left_border_columns_pln_pln(Rpp8u **srcPtrTemp, Rpp8u *dstPtrTemp, Rpp32u kernelSize, Rpp32u padLength,
                                                Rpp32u unpaddedWidth, Rpp32s rowKernelLoopLimit, Rpp32f *filterXTensor, Rpp32f *filterYTensor)
{
    for (int k = 0; k < padLength; k++)
    {
        sobel_filter_generic_tensor(srcPtrTemp, dstPtrTemp, k, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterXTensor, filterYTensor, 1);
        dstPtrTemp++;
    }
}

Rpp32f sobel3x3X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
Rpp32f sobel3x3Y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

RppStatus sobel_filter_u8_u8_host_tensor(Rpp8u *srcPtr,
                                         RpptDescPtr srcDescPtr,
                                         Rpp8u *dstPtr,
                                         RpptDescPtr dstDescPtr,
                                         Rpp32u sobelType,
                                         Rpp32u kernelSize,
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
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u padLength = kernelSize / 2;
        Rpp32u bufferLength = roi.xywhROI.roiWidth * layoutParams.bufferMultiplier;
        Rpp32u unpaddedHeight = roi.xywhROI.roiHeight - padLength;
        Rpp32u unpaddedWidth = roi.xywhROI.roiWidth - padLength;
        bool combined = (sobelType == 2);
        Rpp32f *filter, *filterX, *filterY;

        Rpp8u *srcPtrChannel, *dstPtrChannel;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        if (kernelSize == 3)
        {
            __m256 pFilter[9], pFilterX[9], pFilterY[9];
            if (sobelType == 0)
            {
                filter = sobel3x3X;
                for (int i = 0; i < 9; i++)
                    pFilter[i] = _mm256_set1_ps(filter[i]);
            }
            else if (sobelType == 1)
            {
                filter = sobel3x3Y;
                for (int i = 0; i < 9; i++)
                    pFilter[i] = _mm256_set1_ps(filter[i]);
            }
            else
            {
                filterX = sobel3x3X;
                filterY = sobel3x3Y;
                for (int i = 0; i < 9; i++)
                {
                    pFilterX[i] = _mm256_set1_ps(filterX[i]);
                    pFilterY[i] = _mm256_set1_ps(filterY[i]);
                }
            }

            Rpp8u *srcPtrRow[3], *dstPtrRow;
            for (int i = 0; i < 3; i++)
                srcPtrRow[i] = srcPtrChannel + i * srcDescPtr->strides.hStride;
            dstPtrRow = dstPtrChannel;

            // box filter without fused output-layout toggle (NCHW -> NCHW)
            if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW) && (srcDescPtr->c == 1))
            {
                /* exclude 2 * padLength number of columns from alignedLength calculation
                   since padLength number of columns from the beginning and end of each row will be computed using raw c code */
                Rpp32u alignedLength = ((bufferLength - (2 * padLength)) / 8) * 8;
                for (int c = 0; c < srcDescPtr->c; c++)
                {
                    srcPtrRow[0] = srcPtrChannel;
                    srcPtrRow[1] = srcPtrRow[0] + srcDescPtr->strides.hStride;
                    srcPtrRow[2] = srcPtrRow[1] + srcDescPtr->strides.hStride;
                    dstPtrRow = dstPtrChannel;
                    for(int i = 0; i < roi.xywhROI.roiHeight; i++)
                    {
                        int vectorLoopCount = 0;
                        bool padLengthRows = (i < padLength) ? 1: 0;
                        Rpp8u *srcPtrTemp[3] = {srcPtrRow[0], srcPtrRow[1], srcPtrRow[2]};
                        Rpp8u *dstPtrTemp = dstPtrRow;

                        // get the number of rows needs to be loaded for the corresponding row
                        Rpp32s rowKernelLoopLimit = kernelSize;
                        get_kernel_loop_limit(i, rowKernelLoopLimit, padLength, unpaddedHeight);
                        if (!combined)
                            process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                        else
                            process_left_border_columns_pln_pln(srcPtrTemp, dstPtrTemp, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);

                        dstPtrTemp += padLength;
#if __AVX2__
                        // process alignedLength number of columns in each row
                        for (; vectorLoopCount < alignedLength; vectorLoopCount += 8)
                        {
                            __m256 pRow[6];
                            // irrespective of row location, we need to load 2 rows for 3x3 kernel
                            rpp_load16_u8_to_f32_avx(srcPtrTemp[0], &pRow[0]);
                            rpp_load16_u8_to_f32_avx(srcPtrTemp[1], &pRow[2]);

                            // if rowKernelLoopLimit is 3 load values from 3rd row pointer else set it 0
                            if (rowKernelLoopLimit == 3)
                                rpp_load16_u8_to_f32_avx(srcPtrTemp[2], &pRow[4]);
                            else
                            {
                                pRow[4] = avx_p0;
                                pRow[5] = avx_p0;
                            }

                            __m256 pDst[2];
                            if (!combined)
                            {
                                pDst[0] = avx_p0;
                                pDst[1] = avx_p0;
                                for (int k = 0; k < 3; k++)
                                {
                                    __m256 pTemp[3];
                                    Rpp32s filterIndex =  k * 3;
                                    Rpp32s rowIndex = k * 2;

                                    pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilter[filterIndex]);
                                    pTemp[1] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1), pFilter[filterIndex + 1]);
                                    pTemp[2] = _mm256_mul_ps(_mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2), pFilter[filterIndex + 2]);
                                    pDst[0] = _mm256_add_ps(pDst[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                                }
                            }
                            else
                            {
                                __m256 pDstX[2], pDstY[2];
                                for (int k = 0; k < 2; k++)
                                {
                                    pDstX[k] = avx_p0;
                                    pDstY[k] = avx_p0;
                                    pDst[k] = avx_p0;
                                }
                                for (int k = 0; k < 3; k++)
                                {
                                    __m256 pTemp[3], pRowShift[2];
                                    Rpp32s filterIndex =  k * 3;
                                    Rpp32s rowIndex = k * 2;

                                    pRowShift[0] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 1), avx_pxMaskRotate0To1);
                                    pRowShift[1] = _mm256_permutevar8x32_ps(_mm256_blend_ps(pRow[rowIndex], pRow[rowIndex + 1], 3), avx_pxMaskRotate0To2);
                                    pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterX[filterIndex]);
                                    pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterX[filterIndex + 1]);
                                    pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterX[filterIndex + 2]);
                                    pDstX[0] = _mm256_add_ps(pDstX[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));

                                    pTemp[0] = _mm256_mul_ps(pRow[rowIndex], pFilterY[filterIndex]);
                                    pTemp[1] = _mm256_mul_ps(pRowShift[0], pFilterY[filterIndex + 1]);
                                    pTemp[2] = _mm256_mul_ps(pRowShift[1], pFilterY[filterIndex + 2]);
                                    pDstY[0] = _mm256_add_ps(pDstY[0], _mm256_add_ps(_mm256_add_ps(pTemp[0], pTemp[1]), pTemp[2]));
                                }
                                pDstX[0] = _mm256_mul_ps(pDstX[0], pDstX[0]);
                                pDstY[0] = _mm256_mul_ps(pDstY[0], pDstY[0]);
                                pDst[0] =  _mm256_sqrt_ps(_mm256_add_ps(pDstX[0], pDstY[0]));
                            }
                            rpp_store16_f32_to_u8_avx(dstPtrTemp, pDst);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 8);
                            dstPtrTemp += 8;
                        }
#endif
                        vectorLoopCount += padLength;
                        for (; vectorLoopCount < bufferLength; vectorLoopCount++)
                        {
                            if (!combined)
                                sobel_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filter);
                            else
                                sobel_filter_generic_tensor(srcPtrTemp, dstPtrTemp, vectorLoopCount, kernelSize, padLength, unpaddedWidth, rowKernelLoopLimit, filterX, filterY);
                            increment_row_ptrs(srcPtrTemp, kernelSize, 1);
                            dstPtrTemp++;
                        }
                        // for the first padLength rows, we need not increment the src row pointers to next rows
                        increment_row_ptrs(srcPtrRow, kernelSize, (!padLengthRows) ? srcDescPtr->strides.hStride : 0);
                        dstPtrRow += dstDescPtr->strides.hStride;
                    }
                    srcPtrChannel += srcDescPtr->strides.cStride;
                    dstPtrChannel += dstDescPtr->strides.cStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}