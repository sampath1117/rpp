#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

void compute_diff_square_sum(Rpp32f &output, Rpp32f *input, Rpp64s inputStride, Rpp64s numElements, Rpp32f mean)
{
    const Rpp64s stride = 1;
    if (numElements > 32)
    {
        Rpp64s currElements = numElements >> 1;
        Rpp32f tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_diff_square_sum(tmp1, input, stride, currElements, mean);

        // reduce second half and accumulate
        compute_diff_square_sum(tmp2, input + currElements * stride, stride, numElements - currElements, mean);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        Rpp32f tmp = 0;
        for (Rpp64s i = 0; i < numElements; i++)
        {
            Rpp32f curr = (input[i * stride] - mean);
            auto curnew = curr * curr;
            tmp += curnew;
        }

        // accumulate in target value
        output += tmp;
    }
}

void compute_sum(Rpp32f &output, Rpp32f *input, Rpp64s inputStride, Rpp64s numElements)
{
    const Rpp64s stride = 1;
    if (numElements > 32)
    {
        Rpp64s currElements = numElements >> 1;
        Rpp32f tmp1 = 0, tmp2 = 0;

        // reduce first half and accumulate
        compute_sum(tmp1, input, stride, currElements);

        // reduce second half and accumulate
        compute_sum(tmp2, input + currElements * stride, stride, numElements - currElements);

        tmp1 += tmp2;
        output += tmp1;
    }
    else
    {
        // reduce to a temporary
        Rpp32f tmp = 0;
        for (Rpp64s i = 0; i < numElements; i++)
            tmp += input[i * stride];

        // accumulate in target value
        output += tmp;
    }
}

Rpp32f rpp_rsqrt(Rpp32f x)
{
    // Use SSE intrinsic and one Newton-Raphson refinement step
    // - faster and less hacky than the hack below.
    __m128 X = _mm_set_ss(x);
    __m128 tmp = _mm_rsqrt_ss(X);
    Rpp32f y = _mm_cvtss_f32(tmp);
    return y * (1.5f - x * 0.5f * y * y);
}

static void rpp_rsqrt_sse(Rpp32f *input, Rpp64s numElements, Rpp32f eps, Rpp32f rdiv, Rpp32f mul)
{
    Rpp64s i = 0;
    __m128 rdivx4 = _mm_set1_ps(rdiv);
    __m128 mulx4 = _mm_set1_ps(mul * 0.5f);
    if (eps) // epsilon is present - no need for masking, but we need to add it
    {
        __m128 epsx4 = _mm_set1_ps(eps);
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            x = _mm_add_ps(x, epsx4);
            __m128 y = _mm_rsqrt_ps(x);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    else
    {
        for (; i + 4 <= numElements; i += 4)
        {
            __m128 x = _mm_loadu_ps(&input[i]);
            x = _mm_mul_ps(x, rdivx4);
            __m128 mask = _mm_cmpneq_ps(x, xmm_p0);
            __m128 y = _mm_rsqrt_ps(x);
            y = _mm_and_ps(y, mask);
            __m128 y2 = _mm_mul_ps(y, y);
            __m128 xy2 = _mm_mul_ps(x, y2);
            __m128 three_minus_xy2 = _mm_sub_ps(xmm_p3, xy2);
            y = _mm_mul_ps(y, three_minus_xy2);
            y = _mm_mul_ps(y, mulx4);
            _mm_storeu_ps(&input[i], y);
        }
    }
    if (eps)
    {
        for (; i < numElements; i++)
            input[i] = rpp_rsqrt(input[i] * rdiv + eps) * mul;
    }
    else
    {
        for (; i < numElements; i++)
        {
            Rpp32f x = input[i] * rdiv;
            input[i] = x ? rpp_rsqrt(x) * mul : 0;
        }
    }
}

void compute_2D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = 1.0 / dims[1];
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        meanPtr[i] = 0;
        compute_sum(meanPtr[i], srcPtrTemp, 1, dims[1]);
        srcPtrTemp += stride[1];
        meanPtr[i] = meanPtr[i] * normFactor;
    }
}

void compute_2D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {

    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f normFactor = (Rpp32f)(1.0 / dims[1]);
    for(Rpp32u i = 0; i < dims[0]; i++)
    {
        stdDevPtr[i] = 0;
        compute_diff_square_sum(stdDevPtr[i], srcPtrTemp, 1, dims[1], meanPtr[i]);
        srcPtrTemp += stride[1];
    }
    rpp_rsqrt_sse(stdDevPtr, (Rpp64s)dims[0], 0, normFactor, 1);
}

void normalize_2D_tensor(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    for(Rpp32u i = 0; i < dims[0]; i++) {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;
        for(Rpp32u j = 0; j < dims[1]; j++) {
            *dstPtrTempRow++ = (*srcPtrTempRow++ - meanPtr[paramIdx]) * invStdDevPtr[paramIdx] + shift;
            paramIdx += paramStride[0];
        }
        paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
        srcPtrTemp += (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
        dstPtrTemp += (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? dstDescPtr->strides.hStride : dstDescPtr->strides.wStride;
    }
}

void normalize_2D_tensor_avx_axis2(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                                   Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    Rpp32u vectorIncrement = 8;
    Rpp32u bufferLength = dims[1];
    Rpp32u alignedLength = (bufferLength / 8) * 8;
    Rpp32u numRows = dims[0];

    __m256 pShift = _mm256_set1_ps(shift);
    for(Rpp32u i = 0; i < numRows; i++)
    {
        Rpp32f *srcPtrTempRow = srcPtrTemp + i * srcDescPtr->strides.hStride;
        Rpp32f *dstPtrTempRow = dstPtrTemp + i * dstDescPtr->strides.hStride;

        // set mean and stddev
        Rpp32f mean = meanPtr[i];
        Rpp32f invStdDev = invStdDevPtr[i];
        __m256 pMean, pInvStdDev;
        pMean = _mm256_set1_ps(mean);
        pInvStdDev = _mm256_set1_ps(invStdDev);

        Rpp32u vectorLoopCount = 0;
        for(; vectorLoopCount < alignedLength ; vectorLoopCount += 8)
        {
            __m256 pSrc = _mm256_loadu_ps(srcPtrTempRow);
            __m256 pDst = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(pSrc, pMean), pInvStdDev), pShift);
            _mm256_storeu_ps(dstPtrTempRow, pDst);
            srcPtrTempRow += 8;
            dstPtrTempRow += 8;
        }
        for(; vectorLoopCount < dims[1] ; vectorLoopCount += 8)
             *dstPtrTempRow++ = (*srcPtrTempRow++ - mean) * invStdDev + shift;
    }
}

RppStatus normalize_audio_host_tensor(Rpp32f* srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcLengthTensor,
                                      Rpp32s *channelsTensor,
                                      Rpp32s axis_mask,
                                      Rpp32f mean,
                                      Rpp32f stdDev,
                                      Rpp32f scale,
                                      Rpp32f shift,
                                      Rpp32f epsilon,
                                      Rpp32s ddof,
                                      Rpp32u numOfDims)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(8)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // Set all values in dst buffer to 0.0
        for(int cnt = 0; cnt < dstDescPtr->strides.nStride; cnt++)
            dstPtrTemp[cnt] = 0.0f;

        Rpp32u srcAudioDims[numOfDims], srcReductionDims[numOfDims], srcStride[numOfDims], paramStride[numOfDims];
        srcAudioDims[0] = srcLengthTensor[batchCount];
        srcAudioDims[1] = channelsTensor[batchCount];
        if (axis_mask == 3) {
            srcStride[0] = srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = 1;
            srcReductionDims[1] = srcAudioDims[0] * srcAudioDims[1];
            paramStride[0] = paramStride[1] = 0;
        } else if (axis_mask == 1) {
            srcStride[0] = (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
            srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = srcAudioDims[1];
            srcReductionDims[1] = srcAudioDims[0];
            paramStride[0] = 1;
            paramStride[1] = 0;
        } else if (axis_mask == 2) {
            srcStride[0] = srcDescPtr->strides.cStride;
            srcStride[1] = (dstDescPtr->h > 1 and dstDescPtr->w > 1) ? srcDescPtr->strides.hStride : srcDescPtr->strides.wStride;
            srcReductionDims[0] = srcAudioDims[0];
            srcReductionDims[1] = srcAudioDims[1];
            paramStride[0] = 0;
            paramStride[1] = 1;
        }

        Rpp32f* meanTensor = (Rpp32f *)malloc(srcReductionDims[0] * sizeof(Rpp32f));
        Rpp32f* stdDevTensor = (Rpp32f *)malloc(srcReductionDims[0] * sizeof(Rpp32f));

        meanTensor[0] = mean;
        stdDevTensor[0] = stdDev;

        if(!mean)
            compute_2D_mean(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
        if(!stdDev)
            compute_2D_inv_std_dev(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);

        // Inv std dev calculations missing
        if(axis_mask == 2)
            normalize_2D_tensor_avx_axis2(srcPtrTemp, srcDescPtr, dstPtrTemp, dstDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);
        else
            normalize_2D_tensor(srcPtrTemp, srcDescPtr, dstPtrTemp, dstDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);

        // No mean and std dev
        // No mean
        // No std dev
        // Mean and std dev
        // Axis : 0
        // Axis : 1
        // if(mean && stdDev)
        // {
        //     for(int d = 0; d < numOfDims; d++)
        //     {
        //         #pragma omp simd
        //         for(int i = 0; i < srcAudioDims[d]; i++)
        //         {
        //             *dstPtrTemp = (*srcPtrTemp - mean) * stdDev + shift;
        //         }
        //     }
        //     return RPP_SUCCESS;
        // }
        // if(!mean)
        // {
        //     for(int d = 0; d < numOfDims; d++)
        //     {
        //         #pragma omp simd
        //         for(int i = 0; i < srcAudioDims[d]; i++)
        //         {
        //             *dstPtrTemp = (*srcPtrTemp - mean) * stdDev + shift;
        //         }
        //     }
        //     return RPP_SUCCESS;
        // }
        free(meanTensor);
        free(stdDevTensor);
    }
    return RPP_SUCCESS;
}