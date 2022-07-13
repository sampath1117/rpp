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
// *param[in] srcDesc source tensor descriptor
// *param[in] srcSize source audio buffer length
// *param[out] detectedIndex beginning index of non silent region
// *param[out] detectionLength length of non silent region
// *param[in] cutOffDB threshold(dB) below which the signal is considered silent
// *param[in] windowLength size of the sliding window used to calculate of the short-term power of the signal
// *param[in] referencePower reference power that is used to convert the signal to dB.
// *param[in] resetInterval number of samples after which the moving mean average is recalculated to avoid loss of precision
// *param[in] referenceMax bool value to specify to use referencePower or not
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_non_silent_region_detection_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, Rpp32s *srcSize, Rpp32s *detectedIndex, Rpp32s *detectionLength, Rpp32f *cutOffDB, Rpp32s *windowLength, Rpp32f *referencePower, Rpp32s *resetInterval, bool *referenceMax, rppHandle_t handle);

/******************** to_decibels ********************/

// To Decibels augmentation for 1D magnitude buffer

// *param[in] srcPtr source tensor memory
// *param[out] dstPtr destination tensor memory
// *param[in] batchSize number of magnitude values to be processed
// *param[in] cutOffDB  minimum or cut-off ratio in dB
// *param[in] multiplier factor by which the logarithm is multiplied
// *param[in] referenceMagnitude Reference magnitude if not provided maximum value of input used as reference
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_to_decibels_host(RppPtr_t magnitudePtr, RppPtr_t DBPtr, Rpp32u batchSize, Rpp32f cutOffDB = -200.0, Rpp32f multiplier = 10.0, Rpp32f referenceMagnitude = 0.0);
#ifdef GPU_SUPPORT
RppStatus rppt_to_decibels_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, Rpp64s *samplesPerChannelTensor, Rpp32f cutOffDB, Rpp32f multiplier, Rpp32f referenceMagnitude, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** pre_emphasis_filter ********************/

// Pre Emphasis Filter augmentation for 1D audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] srcSize source audio buffer length
// *param[in] coeffTensor preemphasis coefficient
// *param[in] borderType border value policy
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_pre_emphasis_filter_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, Rpp32s *srcSizeTensor, Rpp32f *coeffTensor, RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP);
#ifdef GPU_SUPPORT
RppStatus rppt_pre_emphasis_filter_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, Rpp32s *srcSizeTensor, Rpp32f *coeffTensor, RpptAudioBorderType borderType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** down_mixing ********************/

// Downmix stereo audio buffer to mono audio buffer

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] samplesPerChannelTensor number of samples per channel
// *param[in] channelsTensor number of channels in audio buffer
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_down_mixing_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, Rpp64s *samplesPerChannelTensor, Rpp32s *channelsTensor, bool normalizeWeights = false);

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_AUDIO_H