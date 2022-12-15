#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline Rpp64f Hann(Rpp32f x)
{
    return 0.5 * (1 + std::cos(x * M_PI));
}
struct ResamplingWindow
{
    inline std::pair<Rpp32s, Rpp32s> input_range(Rpp32f x)
    {
        Rpp32s i0 = std::ceil(x) - lobes;
        Rpp32s i1 = std::floor(x) + lobes;
        return {i0, i1};
    }

    inline Rpp32f operator()(Rpp32f x)
    {
        Rpp32f fi = x * scale + center;
        Rpp32s i = std::floor(fi);
        Rpp32f di = fi - i;
        return lookup[i] + di * (lookup[i + 1] - lookup[i]);
    }

    inline __m128 operator()(__m128 x)
    {
        __m128 fi = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
        __m128i i = _mm_cvtps_epi32(fi);
        __m128 fifloor = _mm_cvtepi32_ps(i);
        __m128 di = _mm_sub_ps(fi, fifloor);
        Rpp32s idx[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
        __m128 curr = _mm_setr_ps(lookup[idx[0]], lookup[idx[1]], lookup[idx[2]], lookup[idx[3]]);
        __m128 next = _mm_setr_ps(lookup[idx[0] + 1], lookup[idx[1] + 1], lookup[idx[2] + 1], lookup[idx[3] + 1]);
        return _mm_add_ps(curr, _mm_mul_ps(di, _mm_sub_ps(next, curr)));
    }

    Rpp32f scale = 1, center = 1;
    Rpp32s lobes = 0, coeffs = 0;
    std::vector<Rpp32f> lookup;
};

inline void windowed_sinc(ResamplingWindow &window, Rpp32s coeffs, Rpp32s lobes)
{
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scaleEnvelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.resize(coeffs + 2);
    window.center = ((coeffs - 1) * 0.5f) + 1;
    window.scale = 1 / scale;
    for (int i = 1, iMinusCenter = (1 - window.center); i <= coeffs; i++, iMinusCenter++)
        window.lookup[i] = sinc(iMinusCenter * scale) * Hann(iMinusCenter * scaleEnvelope);
}

RppStatus resample_host_tensor(Rpp32f *srcPtr,
                               RpptDescPtr srcDescPtr,
                               Rpp32f *dstPtr,
                               RpptDescPtr dstDescPtr,
                               Rpp32f *inRateTensor,
                               Rpp32f *outRateTensor,
                               Rpp32s *srcLengthTensor,
                               Rpp32s *channelTensor,
                               Rpp32f quality)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(Rpp32s batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f inRate = inRateTensor[batchCount];
        Rpp32f outRate = outRateTensor[batchCount];
        Rpp32s srcLength = srcLengthTensor[batchCount];
        Rpp32s numChannels = channelTensor[batchCount];

        if(outRate == inRate)
        {
            // No need of Resampling, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcLength * numChannels * sizeof(Rpp32f)));
        }
        else
        {
            ResamplingWindow window;
            Rpp32s lobes = std::round((quality * (0.007 * quality - 0.09) + 3));
            Rpp32s lookupSize = lobes * 64 + 1;
            windowed_sinc(window, lookupSize, lobes);
            Rpp64s outBegin = 0;
            Rpp64s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp64s inPos = 0;
            Rpp64s block = 1 << 10;
            Rpp64f scale = inRate / outRate;
            Rpp32f fScale = scale;

            if(numChannels == 1)
            {
                for (Rpp64s outBlock = outBegin; outBlock < outEnd; outBlock += block)
                {
                    Rpp64s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp64s inBlockFloor = std::floor(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockFloor;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockFloor;

                    for (Rpp64s outPos = outBlock; outPos < blockEnd; outPos++, inPos += fScale)
                    {
                        Rpp32s i0, i1;
                        std::tie(i0, i1) = window.input_range(inPos);
                        if (i0 + inBlockFloor < 0)
                            i0 = -inBlockFloor;
                        if (i1 + inBlockFloor >= srcLength)
                            i1 = srcLength - 1 - inBlockFloor;
                        Rpp32f f = 0;
                        Rpp32s i = i0;

                        __m128 f4 = xmm_p0;
                        Rpp32f x = i - inPos;
                        __m128 x4 = _mm_setr_ps(x, x + 1, x + 2, x + 3);
                        for (; i + 3 <= i1; i += 4)
                        {
                            __m128 w4 = window(x4);
                            f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + i), w4));
                            x4 = _mm_add_ps(x4, xmm_p4);
                        }
                        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
                        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
                        f = _mm_cvtss_f32(f4);

                        x = i - inPos;
                        for (; i <= i1; i++, x++)
                        {
                            Rpp32f w = window(x);
                            f += inBlockPtr[i] * w;
                        }

                        dstPtrTemp[outPos] = f;
                    }
                }
            }
            else
            {
                std::vector<Rpp32f> tmp;
                tmp.resize(numChannels);
                for (Rpp64s outBlock = outBegin; outBlock < outEnd; outBlock += block)
                {
                    Rpp64s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp64s inBlockFloor = std::floor(inBlockRaw);

                    Rpp32f inPos = inBlockRaw - inBlockFloor;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockFloor * numChannels;
                    for (Rpp64s outPos = outBlock; outPos < blockEnd; outPos++, inPos += fScale)
                    {
                        Rpp32s i0, i1;
                        std::tie(i0, i1) = window.input_range(inPos);
                        if (i0 + inBlockFloor < 0)
                            i0 = -inBlockFloor;
                        if (i1 + inBlockFloor >= srcLength)
                            i1 = srcLength - 1 - inBlockFloor;

                        memset(tmp.data(), 0.0f, (size_t)(numChannels * sizeof(Rpp32f)));
                        Rpp32f x = i0 - inPos;
                        Rpp32s ofs0 = i0 * numChannels;
                        Rpp32s ofs1 = i1 * numChannels;

                        for (Rpp32s inOfs = ofs0; inOfs <= ofs1; inOfs += numChannels, x++)
                        {
                            Rpp32f w = window(x);
                            for (Rpp32s c = 0; c < numChannels; c++)
                                tmp[c] += inBlockPtr[inOfs + c] * w;
                        }

                        for (Rpp32s c = 0; c < numChannels; c++)
                            dstPtrTemp[outPos * numChannels + c] = tmp[c];
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}
