#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include<complex>
#define pack 8

Rpp32f reduce_add_ps1(__m256 src) {
    __m256 src_add = _mm256_add_ps(src, _mm256_permute2f128_ps(src, src, 1));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(1, 0, 3, 2)));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(2, 3, 0, 1)));
    Rpp32f *addResult = (Rpp32f *)&src_add;
    return addResult[0];
}

__m256 cosFn(__m256 a_n) {
    float* a = (float*)&a_n;
    for (int i = 0; i < pack; i++) {
        a[i] = std::cos(a[i]);
    }
    return a_n;
}

__m256 sinFn(__m256 a_n) {
    float* a = (float*)&a_n;
    for (int i = 0; i < pack; i++) {
        a[i] = std::sin(a[i]);
    }
    return a_n;
}

void HannWindow(float *output, int N) {
    double a = (2 * M_PI / N);
    int v_n = (!(N % pack)) ? N / pack: (N / pack) + 1;
    __m256 t_n = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
    __m256 nextstep_n = _mm256_set1_ps(pack);
    __m256 accum_n = _mm256_set1_ps(0.5);
    __m256 sub_n = avx_p1;
    __m256 a_n = _mm256_set1_ps(a);
    for (int t = 0; t < v_n; t++) {
        __m256 phase_n = _mm256_mul_ps(a_n, _mm256_add_ps(t_n, accum_n));
        __m256 output_n = _mm256_mul_ps(accum_n, _mm256_sub_ps(sub_n, cosFn(phase_n)));
        _mm256_storeu_ps(output + (t * pack), output_n);
        t_n = _mm256_add_ps(t_n, nextstep_n);
    }
}

int getOutputSize(int length, int windowLength, int windowStep, bool centerWindows) {
    if (!centerWindows)
        length -= windowLength;

    return ((length / windowStep) + 1);
}

int getIdxReflect(int idx, int lo, int hi) {
    if (hi - lo < 2)
        return hi - 1;
    for (;;) {
        if (idx < lo)
            idx = 2 * lo - idx;
        else if (idx >= hi)
            idx = 2 * hi - 2 - idx;
        else
            break;
    }
    return idx;
}

RppStatus spectrogram_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
								  RpptDescPtr dstDescPtr,
                                  Rpp32s *srcLengthTensor,
                                  bool centerWindows,
                                  bool reflectPadding,
                                  Rpp32f *windowFunction,
                                  Rpp32s nfft,
                                  Rpp32s power,
                                  Rpp32s windowLength,
                                  Rpp32s windowStep,
                                  RpptSpectrogramLayout layout)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for (int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        bool vertical = (layout == RpptSpectrogramLayout::FT);

        // Generate hanning window
        std::vector<float> windowFn;
        windowFn.resize(windowLength);
        HannWindow(windowFn.data(), windowLength);

        if(nfft == 0.0f)
            nfft = windowLength;

        Rpp32s numBins = nfft / 2 + 1;
        Rpp32s numWindows = getOutputSize(bufferLength, windowLength, windowStep, centerWindows);
        Rpp32s windowCenterOffset = 0;

        if (centerWindows)
            windowCenterOffset = windowLength / 2;

        std::vector<Rpp32f> windowOutput(numWindows * windowLength);
        for (int64_t w = 0; w < numWindows; w++) {
            int64_t windowStart = w * windowStep - windowCenterOffset;
            if (windowStart < 0 || (windowStart + windowLength) > bufferLength) {
                for (int t = 0; t < windowLength; t++) {
                    int64_t outIdx = (vertical) ? (t * numWindows + w) : (w * windowLength + t);
                    int64_t inIdx = windowStart + t;
                    if (reflectPadding) {
                        inIdx = getIdxReflect(inIdx, 0, bufferLength);
                        windowOutput[outIdx] = windowFn[t] * srcPtrTemp[inIdx];
                    } else {
                        if (inIdx >= 0 && inIdx < bufferLength)
                            windowOutput[outIdx] = windowFn[t] * srcPtrTemp[inIdx];
                        else
                        windowOutput[outIdx] = 0;
                    }
                }
            } else {
                for (int t = 0; t < windowLength; t++) {
                    int64_t outIdx = (vertical) ? (t * numWindows + w) : (w * windowLength + t);
                    int64_t inIdx = windowStart + t;
                    windowOutput[outIdx] = windowFn[t] * srcPtrTemp[inIdx];
                }
            }
        }

        Rpp32u hStride = dstDescPtr->strides.hStride;
        std::vector<Rpp32f> windowOutputTemp(nfft);
        Rpp32s windowOutputTempSize = windowOutputTemp.size();
        Rpp32s alignedLength = (windowOutputTempSize / 8) * 8;
        const __m256 pMinusOne = _mm256_set1_ps(-1.0f);

        for (int w = 0; w < numWindows; w++) {
            for (int i = 0; i < nfft; i++) {
                int64_t inIdx = (vertical) ? (i * numWindows + w) : (w * windowLength + i);
                if (i > windowLength)
                    windowOutputTemp[i] = 0;
                else
                    windowOutputTemp[i] = windowOutput[inIdx];
            }

            // Allocate buffers for fft output
            std::vector<std::complex<Rpp32f>> fftOutput;
            fftOutput.clear();
            fftOutput.reserve(numBins);

            // Compute FFT
            for (int k = 0; k < numBins; k++) {
                Rpp32f real = 0.0f, imag = 0.0f;
                __m256 pReal, pImag;
                pReal = avx_p0;
                pImag = avx_p0;
                __m256 pInd = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
                Rpp32f factor = (2.0f * k * M_PI) / nfft;
                __m256 pFactor = _mm256_set1_ps(factor);
                int i = 0;
                for (; i < alignedLength; i += 8) {
                    __m256 pSrc, pSin, pCos;
                    pSrc = _mm256_loadu_ps(windowOutputTemp.data() + i);
                    sincos_ps(_mm256_mul_ps(pInd, pFactor), &pSin, &pCos);
                    pReal = _mm256_add_ps(pReal, _mm256_mul_ps(pSrc, pCos));
                    pImag = _mm256_add_ps(pImag, _mm256_mul_ps(_mm256_mul_ps(pSrc, pMinusOne), pSin));
                    pInd = _mm256_add_ps(pInd, avx_p8);
                }
                real = reduce_add_ps1(pReal);
                imag = reduce_add_ps1(pImag);
                for(; i < windowOutputTempSize; i++) {
                    Rpp32f x = windowOutputTemp[i];
                    real += x * std::cos(factor * i);
                    imag += -x * std::sin(factor * i);
                }

                fftOutput.push_back({real, imag});
            }

            if (power == 2) {
                // Compute power spectrum
                for (int i = 0; i < numBins; i++) {
                    int64_t outIdx = (vertical) ? (i * hStride + w) : (w * hStride + i);
                    dstPtrTemp[outIdx] = std::norm(fftOutput[i]);
                }
            } else {
                // Compute magnitude spectrum
                for (int i = 0; i < numBins; i++) {
                    int64_t outIdx = (vertical) ? (i * hStride + w) : (w * hStride + i);
                    dstPtrTemp[outIdx] = std::abs(fftOutput[i]);
                }
            }
        }
    }

	return RPP_SUCCESS;
}
