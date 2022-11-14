#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

void HannWindow_hip(float *output, int N) {
  double a = (2 * M_PI / N);
  for (int t = 0; t < N; t++) {
    double phase = a * (t + 0.5);
    output[t] = (0.5 * (1.0 - std::cos(phase)));
  }
}

int getOutputSize_hip(int length, int windowLength, int windowStep, bool centerWindows) {
    if (!centerWindows)
        length -= windowLength;
    return ((length / windowStep) + 1);
}

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

__global__ void spectrogram_tensor(float *srcPtr,
                                        uint2 srcStridesNH,
                                        int *srcLengthTensor,
                                        int *numWindowTensor,
                                        int maxNumWindow,
                                        int numSamples,
                                        int windowCenterOffset,
                                        bool reflectPadding,
                                        float *windowFunction,
                                        float *windowOutput,
                                        int windowLength,
                                        int windowStep)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);

    if (id_y >= numSamples)
        return;

    if (id_x >= numWindowTensor[id_y])
        return;

    float *srcPtrTemp = srcPtr + id_y * srcStridesNH.x;
    int bufferLength = srcLengthTensor[id_y];

    float* windowOutputTemp = windowOutput + (id_y * windowLength * maxNumWindow) + (id_x * windowLength);
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

__global__ void fft_tensor(float *dstPtr,
                                uint2 dstStridesNH,
                                int *numWindowTensor,
                                int maxNumWindow,
                                int numSamples,
                                bool reflectPadding,
                                float *windowOutput,
                                int nfft,
                                int numBins,
                                int power,
                                int windowLength,
                                bool vertical)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
    int id_z = (hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z);


    if (id_y >= numSamples)
        return;

    if (id_x >= numWindowTensor[id_y])
        return;

    if (id_z >= numBins)
        return;

    float *dstPtrTemp = dstPtr + id_y * dstStridesNH.x;

    unsigned int hStride = dstStridesNH.y;
    // Compute FFT
    float* windowOutputTemp = windowOutput + (id_y * windowLength * maxNumWindow) + (id_x * windowLength);
    float real = 0.0f, imag = 0.0f;
    float factor = (2.0f * id_z * M_PI) / nfft;
    for(int i = 0 ; i < nfft; i++) {
        float x = *windowOutputTemp++;
        real += x * cosf(factor*i);
        imag += -x * sinf(factor*i);
    }
    int64_t outIdx = (vertical) ? (id_z * hStride + id_x) : (id_x * hStride + id_z);
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
    // auto t_start_1 = std::chrono::high_resolution_clock::now();
    int* numWindowsSrcPtr = (int*)calloc(srcDescPtr->n,sizeof(int));
    int maxNumWindow = -1;

    for (int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32s bufferLength = srcLengthTensor[batchCount];
        numWindowsSrcPtr[batchCount] = getOutputSize_hip(bufferLength, windowLength, windowStep,centerWindows);
        maxNumWindow = maxNumWindow < numWindowsSrcPtr[batchCount] ? numWindowsSrcPtr[batchCount] : maxNumWindow;
    }

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = maxNumWindow;
    int globalThreads_y = srcDescPtr->n;
    int globalThreads_z = 1;


    bool vertical = (layout == RpptSpectrogramLayout::FT);
    int windowCenterOffset = 0;

    if (centerWindows)
        windowCenterOffset = windowLength / 2;

    // auto t_start_2 = std::chrono::high_resolution_clock::now();
    if (windowFunction == NULL) {
        // TODO: clear when necessary
       windowFunction = (float*)calloc(windowLength,sizeof(float));
       HannWindow_hip(windowFunction, windowLength);
    }
    // auto t_end_2 = std::chrono::high_resolution_clock::now();

    if(nfft == 0.0f)
        nfft = windowLength;

    int numBins = nfft / 2 + 1;

    float* windowOutput = (float*)calloc(srcDescPtr->n * maxNumWindow * windowLength,(sizeof(float)));

    // auto t_start_3 = std::chrono::high_resolution_clock::now();
    // auto t_end_3 = std::chrono::high_resolution_clock::now();

    // auto t_start_4 = std::chrono::high_resolution_clock::now();
    float* d_windowOutput, *d_windowFunction;
    hipMalloc(&d_windowOutput, srcDescPtr->n * maxNumWindow * windowLength * sizeof(float));
    hipMalloc(&d_windowFunction,  windowLength * sizeof(float));
    hipMemcpy(d_windowOutput, windowOutput, srcDescPtr->n * maxNumWindow * windowLength * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_windowFunction, windowFunction, windowLength * sizeof(float), hipMemcpyHostToDevice);

    int* d_numWindowsSrcPtr, *d_srcLengthTensor;
    hipMalloc(&d_numWindowsSrcPtr, srcDescPtr->n * sizeof(int));
    hipMalloc(&d_srcLengthTensor, srcDescPtr->n * sizeof(int));
    hipMemcpy(d_numWindowsSrcPtr, numWindowsSrcPtr, srcDescPtr->n * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_srcLengthTensor, srcLengthTensor, srcDescPtr->n * sizeof(int), hipMemcpyHostToDevice);
    // auto t_end_4 = std::chrono::high_resolution_clock::now();

    // auto t_start_5 = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(spectrogram_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       d_srcLengthTensor,
                       d_numWindowsSrcPtr,
                       maxNumWindow,
                       srcDescPtr->n,
                       windowCenterOffset,
                       reflectPadding,
                       d_windowFunction,
                       d_windowOutput,
                       windowLength,
                       windowStep);
    hipDeviceSynchronize();

    localThreads_x = 8;
    localThreads_y = 8;
    localThreads_z = 16;
    globalThreads_x = maxNumWindow;
    globalThreads_y = srcDescPtr->n;
    globalThreads_z = numBins;

    hipLaunchKernelGGL(fft_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       dstPtr,
                       make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
                       d_numWindowsSrcPtr,
                       maxNumWindow,
                       srcDescPtr->n,
                       reflectPadding,
                       d_windowOutput,
                       nfft,
                       numBins,
                       power,
                       windowLength,
                       vertical);
    hipDeviceSynchronize();


    // auto t_end_5 = std::chrono::high_resolution_clock::now();

    // auto t_start_6 = std::chrono::high_resolution_clock::now();
    // auto t_start_61 = std::chrono::high_resolution_clock::now();
    hipFree(d_windowFunction);
    hipFree(d_numWindowsSrcPtr);
    hipFree(d_windowOutput);
    hipFree(d_srcLengthTensor);
    // auto t_end_61 = std::chrono::high_resolution_clock::now();
    // auto t_start_62 = std::chrono::high_resolution_clock::now();
    free(windowFunction);
    free(numWindowsSrcPtr);
    free(windowOutput);
    // auto t_end_62 = std::chrono::high_resolution_clock::now();
    // auto t_end_6 = std::chrono::high_resolution_clock::now();
    // auto t_end_1 = std::chrono::high_resolution_clock::now();
    // // std::cout << "\nTimer 1 (total time): " << std::chrono::duration<double, std::milli>(t_end_1-t_start_1).count()
    //           << " ms\n";
    // std::cout << "\nTimer 2 (hannwindow): " << std::chrono::duration<double, std::milli>(t_end_2-t_start_2).count()
    //           << " ms\n";
    // std::cout << "\nTimer 3 (cos and sin calculation): " << std::chrono::duration<double, std::milli>(t_end_3-t_start_3).count()
    //           << " ms\n";
    // std::cout << "\nTimer 4 (memcpy and malloc): " << std::chrono::duration<double, std::milli>(t_end_4-t_start_4).count()
    //           << " ms\n";
    // std::cout << "\nTimer 5 (kernel launch):" << std::chrono::duration<double, std::milli>(t_end_5-t_start_5).count()
    //           << " ms\n";
    // std::cout << "\nTimer 6 (free): " << std::chrono::duration<double, std::milli>(t_end_6-t_start_6).count()
    //           << " ms\n";
    // std::cout << "\nTimer 61 (device free): " << std::chrono::duration<double, std::milli>(t_end_61-t_start_61).count()
    //           << " ms\n";
    // std::cout << "\nTimer 62 (Host free): " << std::chrono::duration<double, std::milli>(t_end_62-t_start_62).count()
    //           << " ms\n";



    return RPP_SUCCESS;
}