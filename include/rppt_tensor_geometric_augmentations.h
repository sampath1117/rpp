/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
#define RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
#include "rpp.h"
#include "rppdefs.h"
#ifdef __cplusplus
extern "C" {
#endif

/******************** crop ********************/

// Crop augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_crop_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_crop_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT


/******************** crop_mirror_normalize ********************/

// Crop Mirror Normalize augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] offsetTensor offset value for each image in the batch (offsetTensor[n] <= 0)
// *param[in] multiplierTensor multiplier value for each image in the batch (multiplierTensor[n] > 0)
// *param[in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_crop_mirror_normalize_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *offsetTensor, Rpp32f *multiplierTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_crop_mirror_normalize_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *offsetTensor, Rpp32f *multiplierTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT


/******************** warp_affine ********************/

// Warp Affine augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] affineTensor affine matrix values for transformation calculation (2D tensor of size batchSize * 6 for each image in batch)
// *param[in] interpolationType Interpolation type used (RpptInterpolationType::XYWH or RpptRoiType::LTRB)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_warp_affine_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *affineTensor, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_warp_affine_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *affineTensor, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** flip ********************/

// Flip augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] horizontalTensor horizontal flag value to set horizontal flip on/off for each image in the batch (horizontalTensor[n] = 0/1)
// *param[in] verticalTensor vertical flag value to set vertical flip on/off for each image in the batch (verticalTensor[n] = 0/1)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_flip_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_flip_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** resize ********************/

// Resize augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] dstImgSizes destination image size
// *param[in] interpolationType resize interpolation type
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_resize_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_resize_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** resize_mirror_normalize ********************/

// Resize Mirror Normalize augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] dstImgSizes destination image size
// *param[in] interpolationType resize interpolation type
// *param[in] meanTensor mean value for each image in the batch (meanTensor[n] >= 0, 1D tensor of size = batchSize for greyscale images, size = batchSize * 3 for RGB images))
// *param[in] stdDevTensor standard deviation value for each image in the batch (stdDevTensor[n] >= 0, 1D tensor of size = batchSize for greyscale images, size = batchSize * 3 for RGB images)
// *param[in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_resize_mirror_normalize_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_resize_mirror_normalize_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** resize_crop_mirror ********************/

// Resize Crop Mirror augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDesc source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] dstImgSizes destination image size
// *param[in] interpolationType resize interpolation type
// *param[in] mirrorTensor mirror flag value to set mirroring on/off for each image in the batch (mirrorTensor[n] = 0/1)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : successful completion
// *retval RPP_ERROR : Error

RppStatus rppt_resize_crop_mirror_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_resize_crop_mirror_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstImgSizes, RpptInterpolationType interpolationType, Rpp32u *mirrorTensor, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** rotate ********************/

// Rotate augmentation for a NCHW/NHWC layout tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstDescPtr destination tensor descriptor
// *param[in] angle values for image rotation in degrees (positive deg-anticlockwise/negative deg-clockwise)
// *param[in] interpolationType Interpolation type used (RpptInterpolationType::XYWH or RpptRoiType::LTRB)
// *param[in] roiTensorSrc ROI data for each image in source tensor (2D tensor of size batchSize * 4, in either format - XYWH(xy.x, xy.y, roiWidth, roiHeight) or LTRB(lt.x, lt.y, rb.x, rb.y))
// *param[in] roiType ROI type used (RpptRoiType::XYWH or RpptRoiType::LTRB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_rotate_host(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *angle, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_rotate_gpu(RppPtr_t srcPtr, RpptDescPtr srcDescPtr, RppPtr_t dstPtr, RpptDescPtr dstDescPtr, Rpp32f *angle, RpptInterpolationType interpolationType, RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** slice ********************/

/*! \brief Slice augmentation HOST
 * \details This function performs slice augmentation on a generic 4D tensor.
 *          Slice augmentation involves selecting a region of interest (ROI) from the source tensor
 *          and copying it to the destination tensor. Support added for f32 -> f32 and u8 -> u8 dataypes.
 * \param[in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle Host-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_slice_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32s *anchorTensor, Rpp32s *shapeTensor, RppPtr_t fillValue, bool enablePadding, Rpp32u *roiTensor, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Slice augmentation GPU
 * \details This function performs slice augmentation on a generic 4D tensor.
 *          Slice augmentation involves selecting a region of interest (ROI) from the source tensor
 *          and copying it to the destination tensor. Support added for f32 -> f32 and u8 -> u8 dataypes.
 * \param[in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
 * \param [in] rppHandle HIP-handle
 * \return <tt> RppStatus enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref RppStatus</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */
RppStatus rppt_slice_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32s *anchorTensor, Rpp32s *shapeTensor, RppPtr_t fillValue, bool enablePadding, Rpp32u *roiTensor, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** flip_voxel ********************/

// Flip augmentation for a generic 4D Tensor

// *param[in] srcPtr source tensor memory
// *param[in] srcGenericDescPtr source tensor descriptor
// *param[out] dstPtr destination tensor memory
// *param[in] dstGenericDescPtr destination tensor descriptor
// *param[in] horizontalTensor horizontal flag value to set horizontal flip on/off for each image in the batch (horizontalTensor[n] = 0/1)
// *param[in] verticalTensor vertical flag value to set vertical flip on/off for each image in the batch (verticalTensor[n] = 0/1)
// *param[in] depthTensor depth flag value to set vertical flip on/off for each video in the batch (verticalTensor[n] = 0/1)
// *param[in] roiGenericPtrSrc ROI data for each image in source tensor (tensor of batchSize RpptRoiGeneric values)
// *param[in] roiType ROI type used (RpptRoi3DType::XYZWHD or RpptRoi3DType::LTFRBB)
// *param[in] rppHandle HIP-handle for "_gpu" variants and Host-handle for "_host" variants
// *returns a  RppStatus enumeration.
// *retval RPP_SUCCESS : succesful completion
// *retval RPP_ERROR : Error

RppStatus rppt_flip_voxel_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, Rpp32u *depthTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#ifdef GPU_SUPPORT
RppStatus rppt_flip_voxel_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *horizontalTensor, Rpp32u *verticalTensor, Rpp32u *depthTensor, RpptROI3DPtr roiGenericPtrSrc, RpptRoi3DType roiType, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

/******************** transpose_generic ********************/

/*! \brief Transpose augmentation on HOST backend for a nD tensor
 * \details The transpose augmentation performs an input-permutation based transpose on a generic nD tensor.
 * \param[in] srcPtr source tensor in HOST memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr source tensor in HOST memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] permTensor permutation tensor for transpose operation
 * \param[in] roiTensor ROI data for each element in source tensor (tensor of batchSize * number of dimensions * 2 values)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */

RppStatus rppt_transpose_generic_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *permTensor, Rpp32u *roiTensor, rppHandle_t rppHandle);
/*! \brief Normalize Generic HOST
 * \details Normalizes the input generic ND buffer by removing the mean and dividing by the standard deviation for a given ND Tensor.
 *          Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes. Also has toggle variant(NHWC->NCHW) support for 3D.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] axisMask axis along which normalization needs to be done
 * \param[in] meanTensor values to be subtracted from input
 * \param[in] stdDevTensor standard deviation values to scale the input
 * \param[in] computeMean flag to represent internal computation of mean
 * \param[in] computeStddev flag to represent internal computation of stddev
 * \param[in] scale value to be multiplied with data after subtracting from mean
 * \param[in] shift value to be added finally
 * \param[in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */

RppStatus rppt_normalize_generic_host(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u computeMean, Rpp32u computeStddev, Rpp32f scale, Rpp32f shift, Rpp32u *roiTensor, rppHandle_t rppHandle);

#ifdef GPU_SUPPORT
/*! \brief Normalize Generic HOST
 * \details Normalizes the input generic ND buffer by removing the mean and dividing by the standard deviation for a given ND Tensor.
 *          Supports u8->f32, i8->f32, f16->f16 and f32->f32 datatypes. Also has toggle variant(NHWC->NCHW) support for 3D.
 * \param [in] srcPtr source tensor memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr destination tensor memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] axisMask axis along which normalization needs to be done
 * \param[in] meanTensor values to be subtracted from input
 * \param[in] stdDevTensor standard deviation values to scale the input
 * \param[in] computeMean flag to represent internal computation of mean
 * \param[in] computeStddev flag to represent internal computation of stddev
 * \param[in] scale value to be multiplied with data after subtracting from mean
 * \param[in] shift value to be added finally
 * \param[in] roiTensor values to represent dimensions of input tensor
 * \param [in] rppHandle Host-handle
 * \return <tt> Rppt_Status enum</tt>.
 * \returns RPP_SUCCESS <tt>\ref Rppt_Status</tt> on successful completion.
 * Else return RPP_ERROR
 * \ingroup group_tensor_geometric
 */

RppStatus rppt_normalize_generic_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u axisMask, Rpp32f *meanTensor, Rpp32f *stdDevTensor, Rpp32u computeMean, Rpp32u computeStddev, Rpp32f scale, Rpp32f shift, Rpp32u *roiTensor, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef GPU_SUPPORT
/*! \brief Transpose augmentation on HIP backend for a nD tensor
 * \details The transpose augmentation performs an input-permutation based transpose on a generic nD tensor.
 * \param[in] srcPtr source tensor in HIP memory
 * \param[in] srcGenericDescPtr source tensor descriptor
 * \param[out] dstPtr source tensor in HIP memory
 * \param[in] dstGenericDescPtr destination tensor descriptor
 * \param[in] permTensor permutation tensor for transpose operation
 * \param[in] roiGenericPtrSrc ROI data for each element in source tensor (tensor of batchSize RpptRoiGeneric values)
 * \param [in] rppHandle RPP HIP handle created with <tt>\ref rppCreateWithStreamAndBatchSize()</tt>
 * \return A <tt> \ref RppStatus</tt> enumeration.
 * \retval RPP_SUCCESS Successful completion.
 * \retval RPP_ERROR* Unsuccessful completion.
 */
RppStatus rppt_transpose_generic_gpu(RppPtr_t srcPtr, RpptGenericDescPtr srcGenericDescPtr, RppPtr_t dstPtr, RpptGenericDescPtr dstGenericDescPtr, Rpp32u *permTensor, Rpp32u *roiTensor, rppHandle_t rppHandle);
#endif // GPU_SUPPORT

#ifdef __cplusplus
}
#endif
#endif // RPPT_TENSOR_GEOMETRIC_AUGMENTATIONS_H
