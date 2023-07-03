#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline Rpp64f Hann(Rpp64f x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

struct ResamplingWindow {
    inline void input_range(Rpp32f x, Rpp32s *i0, Rpp32s *i1) {
        Rpp32s xc = ceilf(x);
        *i0 = xc - lobes;
        *i1 = xc + lobes;
    }

    inline Rpp32f operator()(Rpp32f x) {
        Rpp32f fi = x * scale + center;
        Rpp32s i = floorf(fi);
        Rpp32f di = fi - i;
        i = std::max(std::min(i, lookup_size - 2), 0);
        Rpp32f curr = lookup[i];
        Rpp32f next = lookup[i + 1];
        return curr + di * (next - curr);
    }

    inline __m128 operator()(__m128 x) {
        __m128 fi = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
        __m128i i = _mm_cvttps_epi32(fi);
        __m128 fifloor = _mm_cvtepi32_ps(i);
        __m128 di = _mm_sub_ps(fi, fifloor);
        int idx[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
        __m128 curr = _mm_setr_ps(lookup[idx[0]],   lookup[idx[1]],
                                lookup[idx[2]],   lookup[idx[3]]);
        __m128 next = _mm_setr_ps(lookup[idx[0]+1], lookup[idx[1]+1],
                                lookup[idx[2]+1], lookup[idx[3]+1]);

        return _mm_add_ps(curr, _mm_mul_ps(di, _mm_sub_ps(next, curr)));
    }

    Rpp32f scale = 1, center = 1;
    Rpp32s lobes = 0, coeffs = 0;
    Rpp32s lookup_size = 0;
    std::vector<Rpp32f> lookup;
};

inline void windowed_sinc(ResamplingWindow &window,
        Rpp32s coeffs, Rpp32s lobes, std::function<Rpp64f(Rpp64f)> envelope = Hann) {
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.clear();
    window.lookup.resize(coeffs + 5);
    window.lookup_size = window.lookup.size();
    Rpp32s center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
        Rpp32f x = (i - center) * scale;
        Rpp32f y = (i - center) * scale_envelope;
        Rpp32f w = sinc(x) * envelope(y);
        window.lookup[i + 1] = w;
    }
    window.center = center + 1;
    window.scale = 1 / scale;
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
#pragma omp parallel for num_threads(8)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32f inRate = inRateTensor[batchCount];
        Rpp32f outRate = outRateTensor[batchCount];
        Rpp32s srcLength = srcLengthTensor[batchCount];
        Rpp32s numChannels = channelTensor[batchCount];

        if(outRate == inRate) {
            // No need of Resampling, do a direct memcpy
            memcpy(dstPtrTemp, srcPtrTemp, (size_t)(srcLength * numChannels * sizeof(Rpp32f)));
        } else {
            ResamplingWindow window;
            Rpp32s lobes = std::round(0.007 * quality * quality - 0.09 * quality + 3);
            Rpp32s lookupSize = lobes * 64 + 1;
            windowed_sinc(window, lookupSize, lobes);
            Rpp32s outBegin = 0;
            Rpp32s outEnd = std::ceil(srcLength * outRate / inRate);
            Rpp32s inPos = 0;
            Rpp32s block = 1 << 8;
            Rpp64f scale = (Rpp64f)inRate / outRate;
            Rpp32f fscale = scale;

            if(numChannels == 1) {
                for (int outBlock = outBegin; outBlock < outEnd; outBlock += block) {
                    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp32s inBlockRounded = std::floor(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f * __restrict__ inBlockPtr = srcPtrTemp + inBlockRounded;

                    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                        Rpp32s i0, i1;
                        window.input_range(inPos, &i0, &i1);
                        if (i0 + inBlockRounded < 0)
                            i0 = -inBlockRounded;
                        if (i1 + inBlockRounded > srcLength)
                            i1 = srcLength - inBlockRounded;
                        Rpp32f f = 0.0f;
                        Rpp32s i = i0;

                        __m128 f4 = xmm_p0;
                        __m128 x4 = _mm_setr_ps(i - inPos, i + 1 - inPos, i + 2 - inPos, i + 3 - inPos);
                        for (; i + 3 < i1; i += 4) {
                            __m128 w4 = window(x4);

                            f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + i), w4));
                            x4 = _mm_add_ps(x4, xmm_p4);
                        }

                        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
                        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
                        f = _mm_cvtss_f32(f4);

                        Rpp32f x = i - inPos;
                        for (; i < i1; i++, x++) {
                            Rpp32f w = window(x);
                            f += inBlockPtr[i] * w;
                        }

                        dstPtrTemp[outPos] = f;
                    }
                }
            }
            else {
                std::vector<Rpp32f> tmp;
                tmp.resize(numChannels);
                for (int outBlock = outBegin; outBlock < outEnd; outBlock += block) {
                    Rpp32s blockEnd = std::min(outBlock + block, outEnd);
                    Rpp64f inBlockRaw = outBlock * scale;
                    Rpp32s inBlockRounded = std::floor(inBlockRaw);

                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f * __restrict__ inBlockPtr = srcPtrTemp + inBlockRounded * numChannels;
                    for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                        Rpp32s i0, i1;
                        window.input_range(inPos, &i0, &i1);
                        if (i0 + inBlockRounded < 0)
                            i0 = -inBlockRounded;
                        if (i1 + inBlockRounded > srcLength)
                            i1 = srcLength - inBlockRounded;

                        for (int c = 0; c < numChannels; c++)
                            tmp[c] = 0;

                        Rpp32f x = i0 - inPos;
                        Rpp32s ofs0 = i0 * numChannels;
                        Rpp32s ofs1 = i1 * numChannels;

                        for (int in_ofs = ofs0; in_ofs < ofs1; in_ofs += numChannels, x++) {
                            Rpp32f w = window(x);
                            for (int c = 0; c < numChannels; c++)
                                tmp[c] += inBlockPtr[in_ofs + c] * w;
                        }

                        for (int c = 0; c < numChannels; c++)
                            dstPtrTemp[outPos * numChannels + c] = tmp[c];
                    }
                }
            }
        }
    }

    return RPP_SUCCESS;
}
