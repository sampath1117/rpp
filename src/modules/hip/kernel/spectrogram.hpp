#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"


__device__ int getIdxReflect(int idx, int lo, int hi) {
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

__global__ void spectrogramTensor(float *srcPtr,
                                        uint2 srcStridesNH,
                                        int *srcLengthTensor,
                                        int maxNumWindow,
                                        int numSamples,
                                        int windowCenterOffset,
                                        bool reflectPadding,
                                        float *windowFunction,
                                        float *windowOutput,
                                        int windowLength,
                                        int windowStep,
                                        bool centerWindows)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = (hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z);

    if (id_z >= numSamples)
        return;

    int numWindow =  ((!centerWindows) ? (srcLengthTensor[id_z] - windowLength) : (srcLengthTensor[id_z])) / windowStep + 1;
    
    if (id_x >= numWindow)
        return;

    float *srcPtrTemp = srcPtr + id_z * srcStridesNH.x;
    int bufferLength = srcLengthTensor[id_z];

    float* windowOutputTemp = windowOutput + (id_z * windowLength * maxNumWindow) + (id_x * windowLength);
    int64_t windowStart = (id_x * windowStep) - windowCenterOffset;
    if (windowStart >= 0 && (windowStart + windowLength) < bufferLength) {
        for (int t = 0; t < windowLength; t++) {
            int64_t inIdx = windowStart + t;
            *windowOutputTemp++ = windowFunction[t] * srcPtrTemp[inIdx];
        }
    } else {
        for (int t = 0; t < windowLength; t++) {
            int64_t inIdx = windowStart + t;

            if (reflectPadding) {
                inIdx = getIdxReflect(inIdx, 0, bufferLength);
                *windowOutputTemp++ = windowFunction[t] * srcPtrTemp[inIdx];
            } else {
                if (inIdx >= 0 && inIdx < bufferLength) {
                    *windowOutputTemp++ = windowFunction[t] * srcPtrTemp[inIdx];
                } else {
                    *windowOutputTemp++ = 0;
                }
            }
        }
    }
}

__global__ void spectrogramTensorHannWindow(float *srcPtr,
                                                uint2 srcStridesNH,
                                                int *srcLengthTensor,
                                                int maxNumWindow,
                                                int numSamples,
                                                int windowCenterOffset,
                                                bool reflectPadding,
                                                float *windowOutput,
                                                int windowLength,
                                                int windowStep,
                                                bool centerWindows)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_z = (hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z);

    double a = (2 * PI / windowLength);

    if (id_z >= numSamples)
        return;

    int numWindow =  ((!centerWindows) ? (srcLengthTensor[id_z] - windowLength) : (srcLengthTensor[id_z])) / windowStep + 1;
    
    if (id_x >= numWindow)
        return;

    float *srcPtrTemp = srcPtr + id_z * srcStridesNH.x;
    int bufferLength = srcLengthTensor[id_z];

    float* windowOutputTemp = windowOutput + (id_z * windowLength * maxNumWindow) + (id_x * windowLength);
    int64_t windowStart = (id_x * windowStep) - windowCenterOffset;
    if (windowStart >= 0 && (windowStart + windowLength) < bufferLength) {
        for (int t = 0; t < windowLength; t++) {
            int64_t inIdx = windowStart + t;
            *windowOutputTemp++ = (0.5 * (1.0 - cosf(a * (t + 0.5)))) * srcPtrTemp[inIdx];
        }
    } else {
        for (int t = 0; t < windowLength; t++) {
            int64_t inIdx = windowStart + t;

            if (reflectPadding) {
                inIdx = getIdxReflect(inIdx, 0, bufferLength);
                *windowOutputTemp++ = (0.5 * (1.0 - cosf(a * (t + 0.5)))) * srcPtrTemp[inIdx];
            } else {
                if (inIdx >= 0 && inIdx < bufferLength) {
                    *windowOutputTemp++ = (0.5 * (1.0 - cosf(a * (t + 0.5)))) * srcPtrTemp[inIdx];
                } else {
                    *windowOutputTemp++ = 0;
                }
            }
        }
    }
}

__global__ void fftTensor(float *dstPtr,
                                uint2 dstStridesNH,
                                int *srcLengthTensor,
                                int maxNumWindow,
                                int numSamples,
                                bool reflectPadding,
                                float *windowOutput,
                                int nfft,
                                int numBins,
                                int power,
                                int windowLength,
                                int windowStep,
                                bool vertical,
                                bool centerWindows)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
    int id_z = (hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z);


    if (id_z >= numSamples)
        return;

    int numWindow =  ((!centerWindows) ? (srcLengthTensor[id_z] - windowLength) : (srcLengthTensor[id_z])) / windowStep + 1;
    
    if (id_x >= numWindow)
        return;

    if (id_y >= numBins)
        return;

    float *dstPtrTemp = dstPtr + id_z * dstStridesNH.x;

    unsigned int hStride = dstStridesNH.y;
    // Compute FFT
    float* windowOutputTemp = windowOutput + (id_z * windowLength * maxNumWindow) + (id_x * windowLength);
    float real = 0.0f, imag = 0.0f;
    float factor = (2.0f * id_y * M_PI) / nfft;
    for(int i = 0 ; i < nfft; i++) {
        float x = *windowOutputTemp++;
        real += x * cosf(factor*i);
        imag += -x * sinf(factor*i);
    }
    int64_t outIdx = (vertical) ? (id_y * hStride + id_x) : (id_x * hStride + id_y);
    dstPtrTemp[outIdx] = (real*real) + (imag*imag);
    if (power == 1) {
        dstPtrTemp[outIdx] = sqrtf(dstPtrTemp[outIdx]);
    }
}


RppStatus hip_exec_spectrogram_tensor(Rpp32f *srcPtr,
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
                                  RpptSpectrogramLayout layout,
                                  rpp::Handle& handle)
{
    int maxNumWindow = -1;

    for (int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        maxNumWindow = std::max(maxNumWindow,  ((!centerWindows) ? (srcLengthTensor[batchCount] - windowLength) : (srcLengthTensor[batchCount])) / windowStep + 1);
    }

    bool vertical = (layout == RpptSpectrogramLayout::FT);
    int windowCenterOffset = 0;

    if (centerWindows)
        windowCenterOffset = windowLength / 2;

    if(nfft == 0.0f)
        nfft = windowLength;

    int numBins = nfft / 2 + 1;
    
    float* d_windowOutput;
    hipMalloc(&d_windowOutput, srcDescPtr->n * maxNumWindow * windowLength * sizeof(float));
    
    int *d_srcLengthTensor;
    hipMalloc(&d_srcLengthTensor, srcDescPtr->n * sizeof(int));
    hipMemcpy(d_srcLengthTensor, srcLengthTensor, srcDescPtr->n * sizeof(int), hipMemcpyHostToDevice);
    
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Z;
    int localThreads_z = LOCAL_THREADS_Y;
    int globalThreads_x = maxNumWindow;
    int globalThreads_y = 1;
    int globalThreads_z = srcDescPtr->n;

    if (windowFunction == NULL) {
        hipLaunchKernelGGL(spectrogramTensorHannWindow,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                        d_srcLengthTensor,
                        maxNumWindow,
                        srcDescPtr->n,
                        windowCenterOffset,
                        reflectPadding,
                        d_windowOutput,
                        windowLength,
                        windowStep,
                        centerWindows);
    } else {
        hipLaunchKernelGGL(spectrogramTensor,
                        dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                        dim3(localThreads_x, localThreads_y, localThreads_z),
                        0,
                        handle.GetStream(),
                        srcPtr,
                        make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                        d_srcLengthTensor,
                        maxNumWindow,
                        srcDescPtr->n,
                        windowCenterOffset,
                        reflectPadding,
                        windowFunction,
                        d_windowOutput,
                        windowLength,
                        windowStep,
                        centerWindows);
    }
    hipDeviceSynchronize();

    localThreads_x = FFT_LOCAL_THREADS_X;
    localThreads_y = FFT_LOCAL_THREADS_Y;
    localThreads_z = FFT_LOCAL_THREADS_Z;
    globalThreads_x = maxNumWindow;
    globalThreads_y = numBins;
    globalThreads_z = srcDescPtr->n;

    hipLaunchKernelGGL(fftTensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       d_srcLengthTensor,
                       maxNumWindow,
                       srcDescPtr->n,
                       reflectPadding,
                       d_windowOutput,
                       nfft,
                       numBins,
                       power,
                       windowLength,
                       windowStep,
                       vertical,
                       centerWindows);
    hipDeviceSynchronize();

    hipFree(d_windowOutput);
    hipFree(d_srcLengthTensor);


    return RPP_SUCCESS;
}