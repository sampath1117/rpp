#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "mel_scale.hpp"

__global__ void melFilterBankKernel(float *srcPtr,
                                    uint3 srcStridesNHH,
                                    float *dstPtr,
                                    uint2 dstStridesNH,
                                    float *weights,
                                    float *normalizeFactor,
                                    int *intervals,
                                    RpptImagePatchPtr bins,
                                    RpptImagePatchPtr dims,
                                    int numFilter)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
    int id_z = (hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z);


    if (id_y >= bins[id_z].width && id_x >= dims[id_z].width)
        return;

    int index = id_z * srcStridesNHH.z + id_y;
    float weightDown = weights[index];
    float weightUp = 1.0f - weightDown;
    int filterUp = intervals[index];
    int filterDown = filterUp - 1;
    
    int srcIdx = id_z * srcStridesNHH.x + id_y * srcStridesNHH.y + id_x;
    int dstIdx = id_z * dstStridesNH.x + id_x;

    if (filterDown >= 0) {
        int dstIdx2 = dstIdx + filterDown * dstStridesNH.y;
        weightDown *= normalizeFactor[id_z * numFilter + filterDown];
        atomicAdd((dstPtr + dstIdx2), (srcPtr[srcIdx] * weightDown));
    }

    if (filterUp >= 0 && filterUp < dims[id_z].width) {
        int dstIdx2 = dstIdx + filterUp * dstStridesNH.y;
        weightUp *= normalizeFactor[id_z * numFilter + filterUp];
        atomicAdd((dstPtr + dstIdx2), (srcPtr[srcIdx] * weightUp));
    }
}

RppStatus hip_exec_mel_filter_bank_tensor(float *srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          float *dstPtr,
                                          RpptDescPtr dstDescPtr,
                                          RpptImagePatchPtr srcDims,
                                          Rpp32f maxFreq,
                                          Rpp32f minFreq,
                                          RpptMelScaleFormula melFormula,
                                          Rpp32s numFilter,
                                          Rpp32f sampleRate,
                                          bool normalize,
                                          rpp::Handle& handle)
{
    BaseMelScale *melScalePtr;
    switch(melFormula) {
        case RpptMelScaleFormula::HTK:
            melScalePtr = new HtkMelScale;
            break;
        case RpptMelScaleFormula::SLANEY:
        default:
            melScalePtr = new SlaneyMelScale();
            break;
    }
    
    if(maxFreq == 0.0f)
        maxFreq = sampleRate / 2;
    
    // Convert lower, higher freqeuncies to mel scale
    Rpp64f melLow = melScalePtr->hz_to_mel(minFreq);
    Rpp64f melHigh = melScalePtr->hz_to_mel(maxFreq);
    Rpp64f melStep = (melHigh - melLow) / (numFilter + 1);
    
    // TODO - Name change
    Rpp32f * batchWeightsDown = (Rpp32f *)malloc(srcDescPtr->n * srcDescPtr->h * sizeof(float));
    Rpp32f * batchNormFactors = (Rpp32f *)malloc(srcDescPtr->n * numFilter * sizeof(float));
    Rpp32s * batchIntervals = (Rpp32s *)malloc(srcDescPtr->n * srcDescPtr->h * sizeof(int));
    RpptImagePatch binDims[srcDescPtr->n];
        
    for(int idx = 0; idx < handle.GetBatchSize(); idx++)
    {
        // Extract nfft, number of Frames, numBins
        Rpp32s nfft = (srcDims[idx].height - 1) * 2;
        Rpp32s numBins = srcDims[idx].height;
        Rpp32s numFrames = srcDims[idx].width;
        
        Rpp64f hzStep = static_cast<Rpp64f>(sampleRate) / nfft;
        Rpp64f invHzStep = 1.0 / hzStep;
        
        Rpp32s fftBinStart = binDims[idx].height = std::ceil(minFreq * invHzStep);
        Rpp32s fftBinEnd = std::floor(maxFreq * invHzStep);
        fftBinEnd = std::min(fftBinEnd, numBins - 1);
        binDims[idx].width = fftBinEnd - fftBinStart + 1;
        
        Rpp32f* weightsDown = batchWeightsDown + idx * srcDescPtr->h;
        Rpp32f *normFactors = batchNormFactors + idx * numFilter;
        Rpp32s *intervals = batchIntervals + idx * srcDescPtr->h;
        
        Rpp32s fftBin = fftBinStart;
        Rpp64f mel0 = melLow, mel1 = melLow + melStep;
        Rpp64f f = fftBin * hzStep;
        
        for (int interval = 0; interval < numFilter + 1; interval++, mel0 = mel1, mel1 += melStep) {
            Rpp64f f0 = melScalePtr->mel_to_hz(mel0);
            Rpp64f f1 = melScalePtr->mel_to_hz(interval == numFilter ? melHigh : mel1);
            Rpp64f slope = 1.0f / (f1 - f0);

            if (normalize && interval < numFilter) {
                Rpp64f f2 = melScalePtr->mel_to_hz(mel1 + melStep);
                normFactors[interval] = 2.0 / (f2 - f0);
            } else { 
                normFactors[interval] = 1.0f;
            }

            for (; fftBin <= fftBinEnd && f < f1; fftBin++, f = fftBin * hzStep) {
                weightsDown[fftBin] = (f1 - f) * slope;
                intervals[fftBin] = interval;
            }
        }
    }
    
    Rpp64f maxInvHzStep = ((srcDescPtr->h - 1) * 2 / (double)sampleRate);
    Rpp32s fftBinStart = std::ceil(minFreq * maxInvHzStep);
    Rpp32s fftBinEnd = std::floor(maxFreq * maxInvHzStep);
    fftBinEnd = std::min(fftBinEnd, ((int)srcDescPtr->h - 1));

    int localThreads_x = 256;
    int localThreads_y = 1;
    int localThreads_z = 1;
    int globalThreads_x = srcDescPtr->w;
    int globalThreads_y = fftBinEnd - fftBinStart + 1;
    int globalThreads_z = srcDescPtr->n;
    
    // Num of frames to be takwn inside kernel!!!
    // fftBinrange and fftBinStart should also be passed

    
    float *d_weightsDown, *d_normFactors;
    int *d_intervals;
    hipMalloc(&d_weightsDown, srcDescPtr->n * srcDescPtr->h * sizeof(float));
    hipMalloc(&d_normFactors, srcDescPtr->n * numFilter * sizeof(float));
    hipMalloc(&d_intervals, srcDescPtr->n * srcDescPtr->h * sizeof(int));
    RpptImagePatch *d_srcDims, *d_binDims;
    hipMalloc(&d_srcDims, srcDescPtr->n * sizeof(RpptImagePatch));
    hipMalloc(&d_binDims, srcDescPtr->n * sizeof(RpptImagePatch));
    
    hipMemcpy(d_weightsDown, batchWeightsDown, srcDescPtr->n * srcDescPtr->h * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_normFactors, batchNormFactors, srcDescPtr->n * numFilter * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_intervals, batchIntervals, srcDescPtr->n * srcDescPtr->h * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_srcDims, srcDims, srcDescPtr->n * sizeof(RpptImagePatch), hipMemcpyHostToDevice);
    hipMemcpy(d_binDims, binDims, srcDescPtr->n * sizeof(RpptImagePatch), hipMemcpyHostToDevice);
    

    hipLaunchKernelGGL(melFilterBankKernel,
            dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
            dim3(localThreads_x, localThreads_y, localThreads_z),
            0,
            handle.GetStream(),
            srcPtr,
            make_uint3(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride, srcDescPtr->h),
            dstPtr,
            make_uint2(dstDescPtr->strides.nStride, dstDescPtr->strides.hStride),
            d_weightsDown,
            d_normFactors,
            d_intervals,
            d_binDims,
            d_srcDims,
            numFilter);
    
    hipDeviceSynchronize();
    hipFree(d_weightsDown);
    hipFree(d_normFactors);
    hipFree(d_intervals);
    hipFree(d_srcDims);
    hipFree(d_binDims);
    free(batchWeightsDown);
    free(batchNormFactors);
    free(batchIntervals);
    return RPP_SUCCESS;
}