#include "rppdefs.h"
#include <omp.h>

Rpp32f getSquare(Rpp32f &value)
{
	Rpp32f res = value;
	return (res * res);
}

Rpp32f getMax(std::vector<float> &values, int length)
{
	Rpp32f max = values[0];
	for(int i = 1; i < length; i++)
	{
		max = std::max(max, values[i]);
	}
	return max;
}

RppStatus non_silent_region_detection_host_tensor(Rpp32f *srcPtr,
												  RpptDescPtr srcDescPtr,
												  Rpp32s *srcSizeTensor,
												  Rpp32s *detectedIndexTensor,
												  Rpp32s *detectionLengthTensor,
												  Rpp32f *cutOffDBTensor,
												  Rpp32s *windowLengthTensor,
												  Rpp32f *referencePowerTensor,
												  Rpp32s *resetIntervalTensor,
												  bool *referenceMaxTensor)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
		Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32s srcSize = srcSizeTensor[batchCount];
		Rpp32s windowLength = windowLengthTensor[batchCount];
		Rpp32f referencePower = referencePowerTensor[batchCount];
		Rpp32f cutOffDB = cutOffDBTensor[batchCount];
		bool referenceMax = referenceMaxTensor[batchCount];

		// set reset interval based on the user input
		Rpp32s resetInterval = resetIntervalTensor[batchCount];
		resetInterval = (resetInterval == -1) ? srcSize : resetInterval;

		// Calculate buffer size for mms array and allocate mms buffer
		Rpp32s mmsBufferSize = srcSize - windowLength + 1;
		std::vector<float> mmsBuffer;
		mmsBuffer.reserve(mmsBufferSize);

		// Calculate moving mean square of input array and store in mms buffer
		Rpp32f sumOfSquares = 0.0f;
		Rpp32f meanFactor = 1.0f / windowLength;
		int windowBegin = 0;
		while(windowBegin <= srcSize - windowLength)
		{
			for(int i = windowBegin; i < windowLength; i++)
				sumOfSquares += getSquare(srcPtrTemp[i]);
			mmsBuffer[windowBegin] = sumOfSquares * meanFactor;

			auto interval_endIdx = std::min(windowBegin + resetInterval, srcSize) - windowLength + 1;
			for(windowBegin++; windowBegin < interval_endIdx; windowBegin++)
			{
				sumOfSquares += getSquare(srcPtrTemp[windowBegin + windowLength - 1]) - getSquare(srcPtrTemp[windowBegin - 1]);
				mmsBuffer[windowBegin] = sumOfSquares * meanFactor;
			}
		}

		// Convert cutOff from DB to magnitude
		Rpp32f base = (referenceMax) ? getMax(mmsBuffer, mmsBufferSize) : referencePower;
		Rpp32f cutOffMag = base * std::pow(10.0f, cutOffDB * 0.1f);

		// Calculate begining index, length of non silent region from the mms buffer
		int endIdx = mmsBufferSize;
		int beginIdx = endIdx;
		for(int i = 0; i < endIdx; i++)
		{
			if(mmsBuffer[i] >= cutOffMag)
			{
				beginIdx = i;
				break;
			}
		}

		if(beginIdx == endIdx)
		{
			detectedIndexTensor[batchCount] = 0;
			detectionLengthTensor[batchCount] = 0;
		}
		else
		{
			for(int i = endIdx - 1; i >= beginIdx; i--)
			{
				if(mmsBuffer[i] >= cutOffMag)
				{
					endIdx = i;
					break;
				}
			}
			detectedIndexTensor[batchCount] = beginIdx;
			detectionLengthTensor[batchCount] = endIdx - beginIdx + 1;
		}

		// Extend non silent region
		if(detectionLengthTensor[batchCount] != 0)
			detectionLengthTensor[batchCount] += windowLength - 1;
	}

	return RPP_SUCCESS;
}
