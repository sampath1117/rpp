#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ int compute_index_2d(int y, int x, uint *paramShape, uint *paramStrides)
{
    int yFactor =  (paramShape[0] > 1) ? (y % paramShape[0]) * paramStrides[0] : 0;
    int xFactor =  (paramShape[1] > 1) ? (x % paramShape[1]) * paramStrides[1] : 0;
    int paramIndex = yFactor + xFactor;
    return paramIndex;
}

__global__ void normalize_2d_hip_tensor(float *input,
                                        uint2 srcStridesNH,
                                        float *output,
                                        uint2 dstStridesNH,
                                        float *meanTensor,
                                        float *stdDevTensor,
                                        uint *roiTensor,
                                        uint *paramShapeTensor,
                                        uint *paramStridesTensor,
                                        uint maxParamVolume)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    uint *roi = &roiTensor[id_z * 4 + 2];
    uint height = roi[0];
    uint width = roi[1];

    if (id_x >= width || id_y >= height)
        return;

    uint *paramShape = &paramShapeTensor[id_z * 2];
    uint *paramStrides = &paramStridesTensor[id_z * 2];
    int paramIndex = compute_index_2d(id_y, id_x, paramShape, paramStrides);

    uint srcIdx = (id_z * srcStridesNH.x) + (id_y * srcStridesNH.y) + id_x;
    uint dstIdx = (id_z * dstStridesNH.x) + (id_y * dstStridesNH.y) + id_x;
    float mean = meanTensor[id_z * maxParamVolume + paramIndex];
    float stdDev = stdDevTensor[id_z * maxParamVolume + paramIndex];
    float invStdDev = 1.0f / stdDev;
    output[dstIdx] = (input[srcIdx] - mean) * invStdDev;
}

void normalize_setup(Rpp32u *roiTensor, Rpp32u batchSize, Rpp32u numDims, Rpp32u axisMask,
                     Rpp32u *paramShapeTensor, Rpp32u *paramStridesTensor, Rpp32u &maxParamVolume)
{
    maxParamVolume = 1;
    for(int i = 0; i < batchSize; i++)
    {
        // calculate the param shape and param volume based on the axis mask
        Rpp32u paramVolume = 1;
        Rpp32u *roi = &roiTensor[numDims * 2 * i + numDims];
        Rpp32u *paramShape = &paramShapeTensor[i * numDims];
        for(int j = 0; j < numDims; j++)
        {
            paramShape[j] = ((axisMask & (int)(pow(2, j))) >= 1) ? 1 : roi[j];
            paramVolume *= paramShape[j];
        }
        maxParamVolume = std::max(maxParamVolume, paramVolume);

        // calculate the param strides from the param shape
        Rpp32u *paramStrides = &paramStridesTensor[i * numDims];
        Rpp32u val = 1;
        for(int j = numDims - 1; j > 0; j--)
        {
            paramStrides[j] = val;
            val *= paramShape[j];
        }
        paramStrides[0] = val;
    }
}

RppStatus hip_exec_normalize_tensor(Rpp32f *srcPtr,
                                    RpptGenericDescPtr srcGenericDescPtr,
                                    Rpp32f *dstPtr,
                                    RpptGenericDescPtr dstGenericDescPtr,
                                    Rpp32u axisMask,
                                    Rpp32f *meanTensor,
                                    Rpp32f *stdDevTensor,
                                    Rpp32u computeMean,
                                    Rpp32u computeStddev,
                                    Rpp32f scale,
                                    Rpp32f shift,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    Rpp32u batchSize = srcGenericDescPtr->dims[0];
    Rpp32u numDims = srcGenericDescPtr->numDims - 1;

    // create buffer for paramShape and paramStride
    Rpp32u *paramShape, *paramStrides;
    hipHostMalloc(&paramShape, batchSize * numDims * sizeof(Rpp32u));
    hipHostMalloc(&paramStrides, batchSize * numDims * sizeof(Rpp32u));

    // do initial preprocessing and fill the values for paramShape and paramStrides
    Rpp32u maxParamVolume;
    normalize_setup(roiTensor, batchSize, numDims, axisMask,
                    paramShape, paramStrides, maxParamVolume);

    // based on number of dimensions call the corresponding kernel
    if (numDims == 2)
    {
        // NHW
        int globalThreads_x = dstGenericDescPtr->dims[2];
        int globalThreads_y = dstGenericDescPtr->dims[1];
        int globalThreads_z = dstGenericDescPtr->dims[0];

        hipLaunchKernelGGL(normalize_2d_hip_tensor,
                           dim3(ceil((float)globalThreads_x/LOCAL_THREADS_X), ceil((float)globalThreads_y/LOCAL_THREADS_Y), ceil((float)globalThreads_z/LOCAL_THREADS_Z)),
                           dim3(LOCAL_THREADS_X, LOCAL_THREADS_Y, LOCAL_THREADS_Z),
                           0,
                           handle.GetStream(),
                           srcPtr,
                           make_uint2(srcGenericDescPtr->strides[0], srcGenericDescPtr->strides[1]),
                           dstPtr,
                           make_uint2(dstGenericDescPtr->strides[0], dstGenericDescPtr->strides[1]),
                           meanTensor,
                           stdDevTensor,
                           roiTensor,
                           paramShape,
                           paramStrides,
                           maxParamVolume);
    }
    else
    {
        // do nothing for now
        int globalThreads_x = dstGenericDescPtr->strides[0];
        int globalThreads_y = 1;
        int globalThreads_z = dstGenericDescPtr->dims[0];
    }

    hipStreamSynchronize(handle.GetStream());
    hipHostFree(paramShape);
    hipHostFree(paramStrides);

    return RPP_SUCCESS;
}