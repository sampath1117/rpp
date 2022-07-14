#include "rppdefs.h"

RppStatus to_decibels_host_tensor(Rpp32f *srcPtr,
                                  RpptDescPtr srcDescPtr,
                                  Rpp32f *dstPtr,
                                  Rpp32u *srcLengthTensor,
                                  Rpp32f cutOffDB,
                                  Rpp32f multiplier,
                                  Rpp32f referenceMagnitude)
{
    bool referenceMax = (referenceMagnitude == 0.0) ? false : true;

    // Calculate the intermediate values needed for DB conversion
    Rpp32f minRatio = pow(10, cutOffDB / multiplier);
    if(minRatio == 0.0f)
        minRatio = std::nextafter(0.0f, 1.0f);

    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32u bufferLength = srcLengthTensor[batchCount];

        // Compute maximum value in the input buffer
        if(!referenceMax)
        {
            referenceMagnitude = srcPtrTemp[0];
            for(int i = 1 ; i < bufferLength ; i++)
            {
                if(srcPtrTemp[i] > referenceMagnitude)
                    referenceMagnitude = srcPtrTemp[i];
            }
        }

        // Avoid division by zero
        if(referenceMagnitude == 0.0)
            referenceMagnitude = 1.0;

        Rpp32f invReferenceMagnitude = 1.f / referenceMagnitude;
        for(int i = 0 ; i < bufferLength ; i++)
            dstPtrTemp[i] = multiplier * log10(std::max(minRatio, srcPtrTemp[i] * invReferenceMagnitude));
    }

    return RPP_SUCCESS;
}