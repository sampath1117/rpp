#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

inline double Hann(double x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

struct ResamplingWindow {
    inline std::pair<int, int> input_range(Rpp32f x) {
        int i0 = std::ceil(x) - lobes;
        int i1 = std::floor(x) + lobes;
        return {i0, i1};
    }

    inline Rpp32f operator()(Rpp32f x) {
        Rpp32f fi = x * scale + center;
        int i = std::floor(fi);
        Rpp32f di = fi - i;
        return lookup[i] + di * (lookup[i + 1] - lookup[i]);
    }

    inline __m128 operator()(__m128 x) {
        __m128 fi = _mm_add_ps(x * _mm_set1_ps(scale), _mm_set1_ps(center));
        __m128i i = _mm_cvtps_epi32(fi);
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
    int lobes = 0, coeffs = 0;
    std::vector<Rpp32f> lookup;
};

inline void windowed_sinc(ResamplingWindow &window,
        int coeffs, int lobes, std::function<double(double)> envelope = Hann) {
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.resize(coeffs + 2);
    int center = (coeffs - 1) * 0.5f;
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
                               Rpp32f quality,
                               size_t internal_batch_size)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(internal_batch_size)
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
            int lobes = std::round(0.007 * quality * quality - 0.09 * quality + 3);
            int lookupSize = lobes * 64 + 1;
            windowed_sinc(window, lookupSize, lobes);
            int64_t outBegin = 0;
            int64_t outEnd = std::ceil(srcLength * outRate / inRate);
            int64_t inPos = 0;
            int64_t block = 1 << 10;
            double scale = inRate / outRate;
            Rpp32f fscale = scale;

            if(numChannels == 1) {
                for (int64_t outBlock = outBegin; outBlock < outEnd; outBlock += block) {
                    int64_t blockEnd = std::min(outBlock + block, outEnd);
                    double inBlockRaw = outBlock * scale;
                    int64_t inBlockRounded = std::floor(inBlockRaw);
                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockRounded;

                    for (int64_t outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                        int i0, i1;
                        std::tie(i0, i1) = window.input_range(inPos);
                        if (i0 + inBlockRounded < 0)
                            i0 = -inBlockRounded;
                        if (i1 + inBlockRounded >= srcLength)
                            i1 = srcLength - 1 - inBlockRounded;
                        Rpp32f f = 0;
                        int i = i0;

                        __m128 f4 = _mm_setzero_ps();
                        __m128 x4 = _mm_setr_ps(i - inPos, i + 1 - inPos, i + 2 - inPos, i + 3 - inPos);
                        for (; i + 3 <= i1; i += 4) {
                            __m128 w4 = window(x4);

                            f4 = _mm_add_ps(f4, _mm_mul_ps(_mm_loadu_ps(inBlockPtr + i), w4));
                            x4 = _mm_add_ps(x4, _mm_set1_ps(4));
                        }

                        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(1, 0, 3, 2)));
                        f4 = _mm_add_ps(f4, _mm_shuffle_ps(f4, f4, _MM_SHUFFLE(0, 1, 0, 1)));
                        f = _mm_cvtss_f32(f4);

                        Rpp32f x = i - inPos;
                        for (; i <= i1; i++, x++) {
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
                for (int64_t outBlock = outBegin; outBlock < outEnd; outBlock += block) {
                    int64_t blockEnd = std::min(outBlock + block, outEnd);
                    double inBlockRaw = outBlock * scale;
                    int64_t inBlockRounded = std::floor(inBlockRaw);

                    Rpp32f inPos = inBlockRaw - inBlockRounded;
                    const Rpp32f *inBlockPtr = srcPtrTemp + inBlockRounded * numChannels;
                    for (int64_t outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale) {
                        int i0, i1;
                        std::tie(i0, i1) = window.input_range(inPos);
                        if (i0 + inBlockRounded < 0)
                            i0 = -inBlockRounded;
                        if (i1 + inBlockRounded >= srcLength)
                            i1 = srcLength - 1 - inBlockRounded;

                        for (int c = 0; c < numChannels; c++)
                            tmp[c] = 0;

                        Rpp32f x = i0 - inPos;
                        int ofs0 = i0 * numChannels;
                        int ofs1 = i1 * numChannels;

                        for (int in_ofs = ofs0; in_ofs <= ofs1; in_ofs += numChannels, x++) {
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
