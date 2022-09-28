#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus remap_u8_u8_host_tensor(Rpp8u *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp8u *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  Rpp32u *rowRemapTable,
                                  Rpp32u *colRemapTable,
                                  RpptROIPtr roiTensorPtrSrc,
                                  RpptRoiType roiType,
                                  RppLayoutParams layoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};

omp_set_dynamic(0);
#pragma omp parallel for num_threads(dstDescPtr->n)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * layoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;
        
        Rpp32u *rowRemapTableImage, *colRemapTableImage;
        rowRemapTableImage = rowRemapTable;
        colRemapTableImage = colRemapTable;

        
        // Remap without fused output-layout toggle (NCHW -> NCHW for 1 channel and 3 channel)
        if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int dstLocRow = 0; dstLocRow < roi.xywhROI.roiHeight; dstLocRow++)
            {
                Rpp8u *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                Rpp32u *rowRemapTableTemp, *colRemapTableTemp;
                rowRemapTableTemp = rowRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);
                colRemapTableTemp = colRemapTableImage + (dstLocRow * roi.xywhROI.roiWidth);

                int vectorLoopCount = 0;
                for (; vectorLoopCount < roi.xywhROI.roiWidth; vectorLoopCount++)
                {
                    Rpp8u * dstPtrTempChannel = dstPtrTemp;
                    Rpp8u * srcPtrTempChannel = srcPtrChannel;
                    for (int c = 0; c < dstDescPtr->c; c++)
                    {
                        *dstPtrTempChannel = *(srcPtrTempChannel + (*rowRemapTableTemp * dstDescPtr->strides.hStride) + *colRemapTableTemp);
                        dstPtrTempChannel += dstDescPtr->strides.cStride;
                        srcPtrTempChannel += srcDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                    rowRemapTableTemp++;
                    colRemapTableTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
            rowRemapTableImage += (roi.xywhROI.roiWidth * roi.xywhROI.roiHeight);
            colRemapTableImage += (roi.xywhROI.roiWidth * roi.xywhROI.roiHeight);
        }
    }

    return RPP_SUCCESS;
}