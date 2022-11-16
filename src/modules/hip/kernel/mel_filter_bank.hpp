#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "mel_scale.hpp"

RppStatus hip_exec_mel_filter_bank_tensor(RppPtr_t srcPtr,
                                          RpptDescPtr srcDescPtr,
                                          RppPtr_t dstPtr,
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
    
    Rpp32s maxNumBins = srcDescPtr->h;
    // TODO - Name change
    Rpp32f batchWeightsDown[handle.GetBatchSize() * srcDescPtr->h];
    Rpp32f batchNormFactors[handle.GetBatchSize() * numFilter];
    Rpp32s batchIntervals[handle.GetBatchSize() * srcDescPtr->h];
        
    for(int idx = 0; idx < handle.GetBatchSize(); idx++)
    {
        // Extract nfft, number of Frames, numBins
        Rpp32s nfft = (srcDims[idx].height - 1) * 2;
        Rpp32s numBins = srcDims[idx].height;
        Rpp32s numFrames = srcDims[idx].width;
        
        Rpp64f hzStep = static_cast<Rpp64f>(sampleRate) / nfft;
        Rpp64f invHzStep = 1.0 / hzStep;
        
        Rpp32s fftBinStart = std::ceil(minFreq * invHzStep);
        Rpp32s fftBinEnd = std::floor(maxFreq * invHzStep);
        fftBinEnd = std::min(fftBinEnd, numBins - 1);
        
        Rpp32f* weightsDown = batchWeightsDown + idx * srcDescPtr->h;
        Rpp32f *normFactors = batchNormFactors + idx * numFilter;
        Rpp32s *intervals = batchIntervals + idx * srcDescPtr->h;
        
        Rpp32s fftBin = fftBinStart;
        Rpp64f mel0 = melLow, mel1 = melLow + melStep;
        Rpp64f f = fftBin * hzStep;
        
        Rpp64f f0 = melScalePtr->mel_to_hz(mel0);
        for (int interval = 0; interval < numFilter + 1; interval++, mel0 = mel1, mel1 += melStep) {
            
            Rpp64f f1 = melScalePtr->mel_to_hz(interval == numFilter ? melHigh : mel1);
            Rpp64f slope = 1.0f / (f1 - f0);

            if (normalize && interval < numFilter) {
                Rpp64f f2 = melScalePtr->mel_to_hz(mel1 + melStep);
                normFactors[interval] = 2.0 / (f2 - f0);
            }

            for (; fftBin <= fftBinEnd && f < f1; fftBin++, f = fftBin * hzStep) {
                weightsDown[fftBin] = (f1 - f) * slope;
                intervals[fftBin] = interval;
            }
        }       
    }
}