#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

RppStatus slice_host_tensor(Rpp32f *srcPtr,
                            RpptDescPtr srcDescPtr,
                            Rpp32f *dstPtr,
                            RpptDescPtr dstDescPtr,
                            Rpp32s *srcDimsTensor,
                            Rpp32f *anchorTensor,
                            Rpp32f *shapeTensor,
                            Rpp32f *fillValues)
{
    omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
    for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
    {
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
        Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32s sampleBatchCount = batchCount * 2;

        // Slice for 1D input
        if (srcDimsTensor[sampleBatchCount + 1] == 1)
        {
            Rpp32s srcBufferLength = srcDimsTensor[sampleBatchCount];
            Rpp32f anchorRaw = anchorTensor[sampleBatchCount];
            Rpp32f shapeRaw = shapeTensor[sampleBatchCount];
            Rpp32f fillValue = fillValues[0];

            Rpp32s anchor = std::llround(anchorRaw);
            Rpp32s shape = std::llround(shapeRaw);

            if (anchor == 0 && shape == srcBufferLength)
            {
            // Do a memcpy if output dimension matches input dimension
                memcpy(dstPtrTemp, srcPtrTemp, shape * sizeof(Rpp32f));
            }
            else
            {
                Rpp32s vectorIncrement = 8;
                Rpp32s alignedLength = shape & ~7;
                __m256 pFillValue = _mm256_set1_ps(fillValue);

                bool needPad = (anchor < 0) || ((anchor + shape) > srcBufferLength);
                Rpp32s dstIdx = 0;
                if (needPad)
                {
                    // out of bounds (left side)
                    Rpp32s numIndices = std::abs(std::min(anchor, 0));
                    Rpp32s leftPadLength = std::min(numIndices, shape);
                    Rpp32s alignedLeftPadLength = leftPadLength & ~7;

                    for (; dstIdx < alignedLeftPadLength; dstIdx += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrTemp, pFillValue);
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; dstIdx < leftPadLength; dstIdx++)
                    {
                        *dstPtrTemp = fillValue;
                        dstPtrTemp++;
                    }

                    anchor += leftPadLength;
                }

                // within input bounds
                Rpp32s srcLengthInBounds = std::max(srcBufferLength - anchor, 0);
                Rpp32s dstLengthInBounds = std::max(shape - dstIdx, 0);
                Rpp32s lengthInBounds = std::min(srcLengthInBounds, dstLengthInBounds);
                memcpy(dstPtrTemp, &srcPtrTemp[anchor], (size_t)(lengthInBounds * sizeof(Rpp32f)));
                dstIdx += lengthInBounds;
                dstPtrTemp += lengthInBounds;

                if (needPad)
                {
                    // out of bounds (right side)
                    for (; dstIdx < alignedLength; dstIdx += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrTemp, pFillValue);
                        dstPtrTemp += vectorIncrement;
                    }
                    for (; dstIdx < shape; dstIdx++)
                    {
                        *dstPtrTemp = fillValue;
                        dstPtrTemp++;
                    }
                }
            }
        }
        else if (srcDimsTensor[sampleBatchCount + 1] > 1)
        {
            Rpp32f anchorRaw[2], shapeRaw[2];
            Rpp32s anchor[2], shape[2];
            anchorRaw[0] = anchorTensor[sampleBatchCount];
            anchorRaw[1] = anchorTensor[sampleBatchCount + 1];
            shapeRaw[0] = shapeTensor[sampleBatchCount];
            shapeRaw[1] = shapeTensor[sampleBatchCount + 1];
            Rpp32f fillValue = fillValues[0];

            anchor[0] = std::llround(anchorRaw[0]);
            shape[0] = std::llround(shapeRaw[0]);
            anchor[1] = std::llround(anchorRaw[1]);
            shape[1] = std::llround(shapeRaw[1]);

            Rpp32s rowBound = std::min(srcDimsTensor[sampleBatchCount], shape[0]);
            Rpp32s colBound = std::min(srcDimsTensor[sampleBatchCount + 1], shape[1]);
            bool needRowPad = (anchor[0] < 0) || ((anchor[0] + shape[0]) > srcDimsTensor[sampleBatchCount]);
            bool needColPad = (anchor[1] < 0) || ((anchor[1] + shape[1]) > srcDimsTensor[sampleBatchCount + 1]);

            Rpp32s vectorIncrement = 8;
            __m256 pFillValue = _mm256_set1_ps(fillValue);
            Rpp32s alignedCol = shape[1] & ~(vectorIncrement - 1);
            Rpp32s alignedColMax = dstDescPtr->strides.hStride & ~(vectorIncrement - 1);

            srcPtrTemp = srcPtrTemp + anchor[0] * srcDescPtr->strides.hStride;
            int row = 0;
            for (; row < rowBound; row++)
            {
                int col = 0;
                Rpp32f *srcPtrRow = srcPtrTemp + row * srcDescPtr->strides.hStride + anchor[1];
                Rpp32f *dstPtrRow = dstPtrTemp;
                memcpy(dstPtrRow, &srcPtrRow[col], colBound * sizeof(Rpp32f));
                col += colBound;
                dstPtrRow += colBound;

                // Fill the columns which are beyond the input width with fill value specified
                if (col < shape[1] && needColPad)
                {
                    for (; col < alignedCol; col += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrRow, pFillValue);
                        dstPtrRow += vectorIncrement;
                    }
                    for (; col < shape[1]; col++)
                        *dstPtrRow++ = fillValue;
                }

                dstPtrTemp += dstDescPtr->strides.hStride;
            }

            // Fill the rows which are beyond the input height with fill value specified
            if (row < shape[0]  && needRowPad)
            {
                for (; row < shape[0]; row++)
                {
                    Rpp32f *dstPtrRow = dstPtrTemp;
                    int vectorLoopCount = 0;
                    for (; vectorLoopCount < alignedColMax; vectorLoopCount += vectorIncrement)
                    {
                        _mm256_storeu_ps(dstPtrRow, pFillValue);
                        dstPtrRow += vectorIncrement;
                    }
                    for (; vectorLoopCount < dstDescPtr->strides.hStride; vectorLoopCount++)
                        *dstPtrRow++ = fillValue;

                    dstPtrTemp += dstDescPtr->strides.hStride;
                }
            }
        }
    }

    return RPP_SUCCESS;
}