#include "rppdefs.h"
#include <omp.h>

RppStatus to_decibels_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  RpptDescPtr dstDescPtr,
                                  RpptImagePatchPtr srcDims,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude,
                                  rpp::Handle& handle)
{
    Rpp32u numThreads = handle.GetNumThreads();
    bool referenceMax = (referenceMagnitude == 0.0) ? false : true;

    // Calculate the intermediate values needed for DB conversion
    Rpp32f minRatio = std::pow(10, cutOffDB / multiplier);
    if(minRatio == 0.0f)
        minRatio = std::nextafter(0.0f, 1.0f);

    Rpp32f log10Factor = 0.3010299956639812;//1 / std::log(10);
    multiplier *= log10Factor;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrCurrent = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrCurrent = dstPtr + batchCount * dstDescPtr->strides.nStride;

        Rpp32u height = srcDims[batchCount].height;
        Rpp32u width = srcDims[batchCount].width;
        Rpp32f refMag = referenceMagnitude;

        // Compute maximum value in the input buffer
        if(!referenceMax)
        {
            refMag = -std::numeric_limits<Rpp32f>::max();
            Rpp32f *srcPtrTemp = srcPtrCurrent;
            if(width == 1)
            {
                refMag = std::max(refMag, *(std::max_element(srcPtrTemp, srcPtrTemp + height)));
            }
            else
            {
                for(int i = 0; i < height; i++)
                {
                    refMag = std::max(refMag, *(std::max_element(srcPtrTemp, srcPtrTemp + width)));
                    srcPtrTemp += srcDescPtr->strides.hStride;
                }
            }
        }

        // Avoid division by zero
        if(refMag == 0.0f)
            refMag = 1.0f;

        Rpp32f invReferenceMagnitude = 1.f / refMag;
        // Interpret as 1D array
        if(width == 1)
        {
            Rpp32s vectorIncrement = 8;
            Rpp32s alignedLength = (height / 8) * 8;
            for(Rpp32s vectorLoopCount = 0; vectorLoopCount < height; vectorLoopCount++)
            {
                *dstPtrCurrent = multiplier * std::log2(std::max(minRatio, (*srcPtrCurrent) * invReferenceMagnitude));
                srcPtrCurrent++;
                dstPtrCurrent++;
            }
        }
        else
        {
            Rpp32s vectorIncrement = 8;
            Rpp32s alignedLength = (width / 8) * 8;

            Rpp32f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrCurrent;
            dstPtrRow = dstPtrCurrent;
            for(int i = 0; i < height; i++)
            {
                Rpp32f *srcPtrTemp, *dstPtrTemp;
                srcPtrTemp = srcPtrRow;
                dstPtrTemp = dstPtrRow;
                Rpp32s vectorLoopCount = 0;
                for(; vectorLoopCount < width; vectorLoopCount++)
                {
                    *dstPtrTemp = multiplier * std::log2(std::max(minRatio, (*srcPtrTemp) * invReferenceMagnitude));
                    srcPtrTemp++;
                    dstPtrTemp++;
                }

                srcPtrRow += srcDescPtr->strides.hStride;
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}