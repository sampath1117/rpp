#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"
#include <chrono>
#include<complex>

Rpp32f reduce_add_ps1(__m256 src) {
    __m256 src_add = _mm256_add_ps(src, _mm256_permute2f128_ps(src, src, 1));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(1, 0, 3, 2)));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(2, 3, 0, 1)));
    Rpp32f *addResult = (Rpp32f *)&src_add;
    return addResult[0];
}

void HannWindow(float *output, int N) {
  double a = (2 * M_PI / N);
  for (int t = 0; t < N; t++) {
    double phase = a * (t + 0.5);
    output[t] = (0.5 * (1.0 - std::cos(phase)));
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
    Rpp32s windowCenterOffset = 0;
    bool vertical = (layout == RpptSpectrogramLayout::FT);

    if (centerWindows)
            windowCenterOffset = windowLength / 2;

    if(nfft == 0.0f)
        nfft = windowLength;

    Rpp32s numBins = nfft / 2 + 1;

    // Generate hanning window
    std::vector<float> windowFn;
    windowFn.resize(windowLength);
    HannWindow(windowFn.data(), windowLength);

    const Rpp32f mul_factor = (2.0f * M_PI) / nfft;

    Rpp32f* cosf = (float*)malloc(sizeof(float) * numBins * nfft);
    Rpp32f* sinf = (float*)malloc(sizeof(float) * numBins * nfft);

    for (int k = 0; k < numBins; k++) {
        for(int i  = 0; i < nfft; i++) {
            cosf[k*nfft+i] = std::cos( k * i * mul_factor);
            sinf[k*nfft+i] = std::sin( k * i * mul_factor);
        }
    }

    Rpp32u hStride = dstDescPtr->strides.hStride;
    Rpp32s alignednfftLength = (nfft / 8) * 8;
    Rpp32s alignednbinsLength = (numBins / 8) * 8;
    Rpp32s alignedwindowLength = (windowLength / 8) * 8;


    if (vertical) {
        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(8)
        for (int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
        {
            Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
            Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
            Rpp32s bufferLength = srcLengthTensor[batchCount];

            Rpp32s numWindows = getOutputSize(bufferLength, windowLength, windowStep, centerWindows);

            Rpp32f* windowOutput = (float*)calloc(numWindows * nfft, sizeof(float));

            for (int64_t w = 0; w < numWindows; w++) {
                int64_t windowStart = w * windowStep - windowCenterOffset;
                Rpp32f* windowOutputTemp = windowOutput + (w * nfft);
                if (windowStart < 0 || (windowStart + windowLength) > bufferLength) {
                    for (int t = 0; t < windowLength; t++) {
                        int64_t inIdx = windowStart + t;
                        if (reflectPadding) {
                            inIdx = getIdxReflect(inIdx, 0, bufferLength);
                            *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                        } else {
                            if (inIdx >= 0 && inIdx < bufferLength)
                                *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                            else
                                *windowOutputTemp++ = 0;
                        }
                    }
                } else {
                    Rpp32f* srcPtrWindowTemp = srcPtrTemp  + windowStart;
                    Rpp32f* windowFnTemp = windowFn.data();
                    int t = 0;
                    for (; t < alignedwindowLength; t += 8) {
                        __m256 pSrc, pWindowFn;
                        pSrc = _mm256_loadu_ps(srcPtrWindowTemp);
                        pWindowFn = _mm256_loadu_ps(windowFnTemp);
                        pSrc = _mm256_mul_ps(pSrc, pWindowFn);
                        _mm256_storeu_ps(windowOutputTemp, pSrc);
                        srcPtrWindowTemp += 8;
                        windowFnTemp += 8;
                        windowOutputTemp += 8;
                    }

                    for (; t < windowLength; t++) {
                        *windowOutputTemp++ = (*windowFnTemp++) * (*srcPtrWindowTemp++);
                    }

                }
            }

            float* fftReal = (float*)calloc(numBins,sizeof(float));
            float* fftImag = (float*)calloc(numBins,sizeof(float));

            for (int64_t w = 0; w < numWindows; w++) {
                // Compute FFT
                for (int k = 0; k < numBins; k++) {
                    float* windowOutputTemp = windowOutput + (w * nfft);
                    float* cosfTemp = cosf  + (k * nfft);
                    float* sinfTemp = sinf  + (k * nfft);
                    float real = 0.0f;
                    float imag = 0.0f;
                    __m256 pReal, pImag;
                    pReal = avx_p0;
                    pImag = avx_p0;
                    int i  = 0;
                    for(; i < alignednfftLength; i += 8) {
                        __m256 pSrc, pSin, pCos;
                        pSrc = _mm256_loadu_ps(windowOutputTemp);
                        pCos = _mm256_loadu_ps(cosfTemp);
                        pSin = _mm256_loadu_ps(sinfTemp);
                        pReal = _mm256_add_ps(pReal, _mm256_mul_ps(pSrc, pCos));
                        pImag = _mm256_add_ps(pImag, _mm256_mul_ps(_mm256_mul_ps(pSrc, avx_pm1), pSin));
                        windowOutputTemp += 8;
                        cosfTemp += 8;
                        sinfTemp += 8;
                    }
                    real = reduce_add_ps1(pReal);
                    imag = reduce_add_ps1(pImag);
                    for (; i < nfft; i++) {
                        float x = *windowOutputTemp++;
                        real += x * *cosfTemp++;
                        imag += -x * *sinfTemp++;
                    }
                    fftReal[k] = real;
                    fftImag[k] = imag;
                }


                Rpp32f* fftRealTemp = fftReal;
                Rpp32f* fftImagTemp = fftImag;
                int i = 0;
                for(; i < alignednbinsLength; i += 8) {
                    __m256 pReal, pImag, pTotal;
                    pReal = _mm256_loadu_ps(fftRealTemp);
                    pImag = _mm256_loadu_ps(fftImagTemp);
                    pReal = _mm256_mul_ps(pReal, pReal);
                    pImag = _mm256_mul_ps(pImag, pImag);
                    pTotal = _mm256_add_ps(pReal,pImag);
                    if (power == 1) {
                        pTotal = _mm256_sqrt_ps(pTotal);
                    }
                    float* pTotalPtr = (float*)&pTotal;
                    for (Rpp32s j = i; j < (i+8); j++) {
                        dstPtrTemp[j * hStride + w] = pTotalPtr[j-i];
                    }
                    fftRealTemp += 8;
                    fftImagTemp += 8;
                }
                for (; i < numBins; i++) {
                    float real = *fftRealTemp++;
                    float imag = *fftImagTemp++;
                    float total = (real*real) + (imag*imag);
                    int64_t outIdx =  (i * hStride + w);
                    if (power == 1) {
                        total = std::sqrt(total);
                    }
                    dstPtrTemp[outIdx]=total;
                }
            }
            free(fftReal);
            free(fftImag);
            free(windowOutput);
        }
    } else {
        float* fftReal = (float*)calloc(srcDescPtr->n*numBins,sizeof(float));
        float* fftImag = (float*)calloc(srcDescPtr->n*numBins,sizeof(float));

        omp_set_dynamic(0);
        #pragma omp parallel for num_threads(8)
        for (int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
        {
            Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
            Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
            Rpp32s bufferLength = srcLengthTensor[batchCount];

            Rpp32s numWindows = getOutputSize(bufferLength, windowLength, windowStep, centerWindows);

            Rpp32f* windowOutput = (float*)calloc(numWindows * nfft, sizeof(float));

            for (int64_t w = 0; w < numWindows; w++) {
                int64_t windowStart = w * windowStep - windowCenterOffset;
                Rpp32f* windowOutputTemp = windowOutput + (w * nfft);

                if (windowStart < 0 || (windowStart + windowLength) > bufferLength) {
                    for (int t = 0; t < windowLength; t++) {
                        int64_t inIdx = windowStart + t;
                        if (reflectPadding) {
                            inIdx = getIdxReflect(inIdx, 0, bufferLength);
                            *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                        } else {
                            if (inIdx >= 0 && inIdx < bufferLength)
                                *windowOutputTemp++ = windowFn[t] * srcPtrTemp[inIdx];
                            else
                                *windowOutputTemp++ = 0;
                        }
                    }
                } else {
                    Rpp32f* srcPtrWindowTemp = srcPtrTemp  + windowStart;
                    Rpp32f* windowFnTemp = windowFn.data();
                    int t = 0;
                    for (; t < alignedwindowLength; t += 8) {
                        __m256 pSrc, pWindowFn;
                        pSrc = _mm256_loadu_ps(srcPtrWindowTemp);
                        pWindowFn = _mm256_loadu_ps(windowFnTemp);
                        pSrc = _mm256_mul_ps(pSrc, pWindowFn);
                        _mm256_storeu_ps(windowOutputTemp, pSrc);
                        srcPtrWindowTemp += 8;
                        windowFnTemp += 8;
                        windowOutputTemp += 8;
                    }

                    for (; t < windowLength; t++) {
                        *windowOutputTemp++ = (*windowFnTemp++) * (*srcPtrWindowTemp++);
                    }
                }

                Rpp32f *fftRealTemp = fftReal + batchCount * numBins;
                Rpp32f *fftImagTemp = fftImag + batchCount * numBins;

                // Compute FFT
                for (int k = 0; k < numBins; k++) {
                    windowOutputTemp = windowOutput + (w * nfft);
                    float* cosfTemp = cosf  + (k * nfft);
                    float* sinfTemp = sinf  + (k * nfft);
                    float real = 0.0f;
                    float imag = 0.0f;
                    __m256 pReal, pImag;
                    pReal = avx_p0;
                    pImag = avx_p0;
                    int i  = 0;
                    for(; i < alignednfftLength; i += 8) {
                        __m256 pSrc, pSin, pCos;
                        pSrc = _mm256_loadu_ps(windowOutputTemp);
                        pCos = _mm256_loadu_ps(cosfTemp);
                        pSin = _mm256_loadu_ps(sinfTemp);
                        pReal = _mm256_add_ps(pReal, _mm256_mul_ps(pSrc, pCos));
                        pImag = _mm256_add_ps(pImag, _mm256_mul_ps(_mm256_mul_ps(pSrc, avx_pm1), pSin));
                        windowOutputTemp += 8;
                        cosfTemp += 8;
                        sinfTemp += 8;
                    }
                    real = reduce_add_ps1(pReal);
                    imag = reduce_add_ps1(pImag);
                    for (; i < nfft; i++) {
                        float x = *windowOutputTemp++;
                        real += x * *cosfTemp++;
                        imag += -x * *sinfTemp++;
                    }
                    *fftRealTemp++ = real;
                    *fftImagTemp++ = imag;
                }

                fftRealTemp = fftReal + batchCount * numBins;
                fftImagTemp = fftImag + batchCount * numBins;

                Rpp32f* dstPtrBinTemp = dstPtrTemp + (w * hStride);
                int i = 0;
                for(; i < alignednbinsLength; i += 8) {
                    __m256 pReal, pImag, pTotal;
                    pReal = _mm256_loadu_ps(fftRealTemp);
                    pImag = _mm256_loadu_ps(fftImagTemp);
                    pReal = _mm256_mul_ps(pReal, pReal);
                    pImag = _mm256_mul_ps(pImag, pImag);
                    pTotal = _mm256_add_ps(pReal,pImag);
                    if (power == 1) {
                        pTotal = _mm256_sqrt_ps(pTotal);
                    }
                    _mm256_storeu_ps(dstPtrBinTemp, pTotal);
                    fftRealTemp += 8;
                    fftImagTemp += 8;
                    dstPtrBinTemp += 8;
                }
                for (; i < numBins; i++) {
                    float real = *fftRealTemp++;
                    float imag = *fftImagTemp++;
                    float total = (real*real) + (imag*imag);
                    if (power == 1) {
                        total = std::sqrt(total);
                    }
                    *dstPtrBinTemp++=total;
                }
            }
            free(windowOutput);
        }
        free(fftReal);
        free(fftImag);
    }
    free(sinf);
    free(cosf);

    return RPP_SUCCESS;
}

