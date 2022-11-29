#include "rppdefs.h"
#include <omp.h>

float hsum_ps(__m128 x)
{
    __m128 shuf = _mm_movehdup_ps(x);        // broadcast elements 3,1 to 2,0
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);        // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

RppStatus down_mixing_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  Rpp32s *channelsTensor,
                                  bool normalizeWeights)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrCurrent = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32s channels = channelsTensor[batchCount];
        Rpp32s samples = srcLengthTensor[batchCount];

        if(channels == 1)
        {
            // No need of downmixing, do a direct memcpy
            memcpy(dstPtrCurrent, srcPtrCurrent, (size_t)(samples * sizeof(Rpp32f)));
        }
        else
        {
            std::vector<float> weights;
            weights.resize(channels, 1.f / channels);
            std::vector<float> normalizedWeights;

            if(normalizeWeights)
            {
                normalizedWeights.resize(channels);

                // Compute sum of the weights
                Rpp32f sum = 0.0;
                for(int i = 0; i < channels; i++)
                    sum += weights[i];

                // Normalize the weights
                Rpp32f invSum = 1.0 / sum;
                for(int i = 0; i < channels; i++)
                    normalizedWeights[i] = weights[i] * invSum;

                weights = normalizedWeights;
            }

            int channelIncrement = 4;
            int alignedChannels = (channels / 4) * 4;

            // use weights to downmix to mono
            Rpp32f *srcPtrRow = srcPtrCurrent;
            for(int64_t dstIdx = 0; dstIdx < samples; dstIdx++)
            {
                Rpp32f *srcPtrTemp = srcPtrRow;
                __m128 pDst = xmm_p0;
                int channelLoopCount = 0;
                for(; channelLoopCount < alignedChannels; channelLoopCount += channelIncrement)
                {
                    __m128 pSrc, pWeights;
                    pWeights = _mm_setr_ps(weights[channelLoopCount], weights[channelLoopCount + 1], weights[channelLoopCount + 2], weights[channelLoopCount + 3]);
                    pSrc = _mm_loadu_ps(srcPtrTemp);
                    pSrc = _mm_mul_ps(pSrc, pWeights);
                    pDst = _mm_add_ps(pDst, pSrc);
                    srcPtrTemp += channelIncrement;
                }
                dstPtrCurrent[dstIdx] = hsum_ps(pDst);
                for(; channelLoopCount < channels; channelLoopCount++)
                {
                    dstPtrCurrent[dstIdx] += ((*srcPtrTemp) * weights[channelLoopCount]);
                    srcPtrTemp++;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}
