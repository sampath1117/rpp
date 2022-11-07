/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RPPT_TENSOR_AUDIO_H
#define RPPT_TENSOR_AUDIO_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** non_silent_region_detection ********************/

// Non Silent Region Detection augmentation for 1D audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[in] srcSize source audio buffer length
// *param[out] detectedIndex beginning index of non silent region
// *param[out] detectionLength length of non silent region
// *param[in] cutOffDB threshold(dB) below which the signal is considered silent
// *param[in] windowLength size of the sliding window used to calculate of the short-term power of the signal
// *param[in] referencePower reference power that is used to convert the signal to dB.
// *param[in] resetInterval number of samples after which the moving mean average is recalculated to avoid loss of precision
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcSize, Rpp32s *detectedIndexTensor, Rpp32s *detectionLengthTensor, Rpp32f cutOffDB, Rpp32s windowLength, Rpp32f referencePower, Rpp32s resetInterval);

/******************** to_decibels ********************/

// To Decibels augmentation for magnitude buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDims source dimensions
// *param[in] cutOffDB  minimum or cut-off ratio in dB
// *param[in] multiplier factor by which the logarithm is multiplied
// *param[in] referenceMagnitude Reference magnitude if not provided maximum value of input used as reference
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_to_decibels_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude);
#ifdef GPU_SUPPORT
RppStatus rppt_to_decibels_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** pre_emphasis_filter ********************/

// Pre Emphasis Filter augmentation for 1D audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcSize source audio buffer length
// *param[in] coeffTensor preemphasis coefficient
// *param[in] borderType border value policy
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcSizeTensor, Rpp32f *coeffTensor, RpptAudioBorderType borderType);
#ifdef GPU_SUPPORT
RppStatus rppt_pre_emphasis_filter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f *coeffTensor, RpptAudioBorderType borderType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** down_mixing ********************/

// Downmix multi channel audio buffer to single channel audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor number of samples per channel
// *param[in] channelsTensor number of channels in audio buffer
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, bool normalizeWeights = false);

/******************** slice ********************/

// Extracts a subtensor or slice from the audio file

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor number of samples per channel
// *param[in] anchor starting index of the slice
// *param[in] shape length of the slice
// *param[in] axes axes along which slice is needed
// *param[in] fillValues fill values based on out of Bound policy
// *param[in] normalized anchor determines whether the anchor positional input should be interpreted as normalized or as absolute coordinates
// *param[in] normalized shape determines whether the shape positional input should be interpreted as normalized or as absolute coordinates
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_slice_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32f *anchorTensor, Rpp32f *shapeTensor, Rpp32s axisMask, Rpp32f *fillValues, bool normalizedAnchor, bool normalizedShape, RpptOutOfBoundsPolicy policyType);

// Mel Filter Bank augmentation

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcDims
// *param[in] maxFreq maximum frequency if not provided maxFreq = sampleRate / 2
// *param[in] minFreq minimum frequency
// *param[in] melFormula
// *param[in] numFilter
// *param[in] sampleRate
// *param[in] normalize
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_mel_filter_bank_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr srcDims, Rpp32f minFreq, Rpp32f maxFreq, RpptMelScaleFormula melFormula, Rpp32s numFilter, Rpp32f sampleRate, bool normalize);

// Spectrogram augmentation

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] srcLengthTensor number of samples per channel
// *param[in] centerWindows
// *param[in] reflectPadding
// *param[in] windowFunction
// *param[in] nfft
// *param[in] power
// *param[in] windowLength
// *param[in] windowStep
// *param[in] layout
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_spectrogram_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, bool centerWindows, bool reflectPadding, Rpp32f *windowFunction, Rpp32s nfft, Rpp32s power, Rpp32s windowLength, Rpp32s windowStep, RpptSpectrogramLayout layout);

/******************** resample ********************/

// Resample audio signal based on the target sample rate

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] inRate
// *param[in] outRate
// *param[in] srcLengthTensor
// *param[in] channelsTensor
// *param[in] quality
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_resample_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *inRateTensor, Rpp32f *outRateTensor, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, Rpp32f quality);

/******************** normalize_audio ********************/

RppStatus rppt_normalize_audio_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32s *srcLengthTensor, Rpp32s *channelsTensor, Rpp32s axisMask,
                                    Rpp32f mean, Rpp32f stdDev, Rpp32f scale, Rpp32f shift, Rpp32f epsilon, Rpp32s ddof, Rpp32s numOfDims);
#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_H