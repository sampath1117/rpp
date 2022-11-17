#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

#define PACK 8

Rpp32f reduce_add_ps2(__m256 src) {
    __m256 src_add = _mm256_add_ps(src, _mm256_permute2f128_ps(src, src, 1));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(1, 0, 3, 2)));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(2, 3, 0, 1)));
    Rpp32f *addResult = (Rpp32f *)&src_add;
    return addResult[0];
}

void compute_2D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    for(Rpp32u i = 0; i < dims[0]; i++) {
        meanPtr[i] = 0;
        int v_n = (!(dims[1] % PACK)) ? dims[1] / PACK: (dims[1] / PACK) + 1;
        __m256 j_n = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        __m256 pack_n = _mm256_set1_ps(PACK);
        __m256 stride_n = _mm256_set1_ps(stride[0]);
        for(Rpp32u j = 0; j < v_n; j++) {
            //meanPtr[i] += (*(srcPtrTemp + j * stride[0]));
            __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
            __m256 meanPtr_n = _mm256_i32gather_ps(srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), 4);
            meanPtr[i] += reduce_add_ps2(meanPtr_n);
            j_n = _mm256_add_ps(j_n, pack_n);
        }
        srcPtrTemp += stride[1];
        meanPtr[i] = meanPtr[i] / dims[1];
    }
}

void compute_2D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    for(Rpp32u i = 0; i < dims[0]; i++) {
        stdDevPtr[i] = 0;
        int v_n = (!(dims[1] % PACK)) ? dims[1] / PACK: (dims[1] / PACK) + 1;
        __m256 j_n = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        __m256 pack_n = _mm256_set1_ps(PACK);
        __m256 stride_n = _mm256_set1_ps(stride[0]);
        __m256 meanptr_n = _mm256_set1_ps(meanPtr[i]);
        for(Rpp32u j = 0; j < v_n; j++) {
            //Rpp32f diff = (*(srcPtrTemp + j * stride[0]) - meanPtr[i]);
            //stdDevPtr[i] += (diff * diff);
            __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
            __m256 diff_n = _mm256_sub_ps(_mm256_i32gather_ps(srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), 4), meanptr_n);
            stdDevPtr[i] += reduce_add_ps2(_mm256_mul_ps(diff_n, diff_n));
            j_n = _mm256_add_ps(j_n, pack_n);
        }
        srcPtrTemp += stride[1];
        stdDevPtr[i] = stdDevPtr[i] / dims[1];
        stdDevPtr[i] = (!stdDevPtr[i]) ? 0.0f : 1.0f / sqrt(stdDevPtr[i]);
    }
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
        srcPtrTemp += srcDescPtr->strides.wStride;
        dstPtrTemp += dstDescPtr->strides.wStride;
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
                                      Rpp32u numOfDims,
                                      size_t internal_batch_size)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(internal_batch_size)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32u srcAudioDims[numOfDims], srcReductionDims[numOfDims], srcStride[numOfDims], paramStride[numOfDims];
        srcAudioDims[0] = srcLengthTensor[batchCount];
        srcAudioDims[1] = channelsTensor[batchCount];

        if (axis_mask == 3) {
            srcStride[0] = srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = 1;
            srcReductionDims[1] = srcAudioDims[0] * srcAudioDims[1];
            paramStride[0] = paramStride[1] = 0;
        } else if (axis_mask == 1) {
            srcStride[0] = srcDescPtr->strides.wStride;
            srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = srcAudioDims[1];
            srcReductionDims[1] = srcAudioDims[0];
            paramStride[0] = 1;
            paramStride[1] = 0;
        } else if (axis_mask == 2) {
            srcStride[0] = srcDescPtr->strides.cStride;
            srcStride[1] = srcDescPtr->strides.wStride;
            srcReductionDims[0] = srcAudioDims[0];
            srcReductionDims[1] = srcAudioDims[1];
            paramStride[0] = 0;
            paramStride[1] = 1;
        }

        Rpp32f *meanTensor = (Rpp32f *)malloc(srcReductionDims[0] * sizeof(Rpp32f));
        Rpp32f *stdDevTensor = (Rpp32f *)malloc(srcReductionDims[0] * sizeof(Rpp32f));

        meanTensor[0] = mean;
        stdDevTensor[0] = stdDev;

        if(!mean)
            compute_2D_mean(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
        if(!stdDev)
            compute_2D_inv_std_dev(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);

        // Inv std dev calculations missing
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