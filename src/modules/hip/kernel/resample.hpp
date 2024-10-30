#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

// -------------------- Set 0 - resample kernel device helpers  --------------------

__device__ __forceinline__ float resample_hip_compute(float &x, float &scale, float &center, float *lookup, int &lookupSize)
{
    float locRaw = x * scale + center;
    int locFloor = std::floor(locRaw);
    float weight = locRaw - locFloor;
    locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
    float current = lookup[locFloor];
    float next = lookup[locFloor + 1];
    return current + weight * (next - current);
}

__device__ __forceinline__ void resample_hip_compute(float4 *src, float4 *dst, const float4 *scale, const float4 *center, float *lookup, int lookupSize)
{
    float4 locRaw = *src * *scale + *center;
    int4 locFloor = make_int4(std::floor(locRaw.x), std::floor(locRaw.y), std::floor(locRaw.z), std::floor(locRaw.w));
    float4 weight = make_float4(locRaw.x - locFloor.x, locRaw.y - locFloor.y, locRaw.z - locFloor.z, locRaw.w - locFloor.w);
    
    // Clamp locFloor values to be within [0, lookupSize - 2]
    locFloor.x = max(min(locFloor.x, lookupSize - 2), 0);
    locFloor.y = max(min(locFloor.y, lookupSize - 2), 0);
    locFloor.z = max(min(locFloor.z, lookupSize - 2), 0);
    locFloor.w = max(min(locFloor.w, lookupSize - 2), 0);

    float4 current = make_float4(lookup[locFloor.x], lookup[locFloor.y], lookup[locFloor.z], lookup[locFloor.w]);
    float4 next = make_float4(lookup[locFloor.x + 1], lookup[locFloor.y + 1], lookup[locFloor.z + 1], lookup[locFloor.w + 1]);
    
    // Perform linear interpolation
    *dst = current + weight * (next - current);
}

// -------------------- Set 2 - resample kernels --------------------

__global__ void resample_single_channel_hip_tensor(float *srcPtr,
                                                   float *dstPtr,
                                                   uint2 strides,
                                                   int2 *srcDimsTensor,
                                                   int *dstDimsTensor,
                                                   RpptResamplingWindow *window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int srcLength = srcDimsTensor[id_z].x;
    int dstLength = dstDimsTensor[id_z];
    int outBlock = id_x;
    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), dstLength);
    double scale = static_cast<double>(srcLength) / dstLength;
    if (outBlock >= dstLength)
        return;

    if (dstLength != srcLength)
    {
        // extract the window scale, center and lookup size values from window
        float windowScale = window->scale;
        float windowCenter = window->center;
        int lookupSize = window->lookupSize;
        float4 windowScale_f4 = static_cast<float4>(windowScale);
        float4 windowCenter_f4 = static_cast<float4>(windowCenter);
        float4 increment_f4 = static_cast<float4>(8.0f);
        d_float8 locInit_f8;
        locInit_f8.f4[0] = make_float4(0.0f, 1.0f, 2.0f, 3.0f);
        locInit_f8.f4[1] = make_float4(4.0f, 5.0f, 6.0f, 7.0f);

        // compute block wise values required for processing
        double inBlockRaw = outBlock * scale;
        int inBlockRounded = floorf(inBlockRaw);
        float inPos = inBlockRaw - inBlockRounded;
        float fscale = scale;
        uint dstIdx = id_z * strides.y + outBlock;
        float *inBlockPtr = srcPtr + id_z * strides.x + inBlockRounded;

        extern __shared__ float lookup_smem[];
        // copy all values from window lookup table to shared memory lookup table
        for (int k = hipThreadIdx_x; k < window->lookupSize; k += hipBlockDim_x)
            lookup_smem[k] = window->lookup[k];
        __syncthreads();

        // process block size (256) elements in single thread
        // for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale, dstIdx++)
        // {
            int loc0, loc1;
            window->input_range(inPos, &loc0, &loc1);

            // check if computed loc0, loc1 values are beyond the input dimensions and update accordingly
            if (loc0 + inBlockRounded < 0)
                loc0 = -inBlockRounded;
            if (loc1 + inBlockRounded > srcLength)
                loc1 = srcLength - inBlockRounded;
            int locInWindow = loc0;
            float locBegin = locInWindow - inPos;
            float accum = 0.0f;

            d_float8 locInWindow_f8, accum_f8;
            locInWindow_f8.f4[0] = static_cast<float4>(locBegin) + locInit_f8.f4[0];
            locInWindow_f8.f4[1] = static_cast<float4>(locBegin) + locInit_f8.f4[1];
            accum_f8.f4[0] = static_cast<float4>(0.0f);
            accum_f8.f4[1] = static_cast<float4>(0.0f);
            for (; locInWindow + 7 < loc1; locInWindow += 8)
            {
                d_float8 weights_f8;
                resample_hip_compute(&locInWindow_f8.f4[0], &weights_f8.f4[0], &windowScale_f4, &windowCenter_f4, lookup_smem, lookupSize);
                resample_hip_compute(&locInWindow_f8.f4[1], &weights_f8.f4[1], &windowScale_f4, &windowCenter_f4, lookup_smem, lookupSize);

                d_float8 src_f8;
                rpp_hip_load8_and_unpack_to_float8(inBlockPtr + locInWindow, &src_f8);
                accum_f8.f4[0] +=  src_f8.f4[0] * weights_f8.f4[0];
                accum_f8.f4[1] +=  src_f8.f4[1] * weights_f8.f4[1];

                locInWindow_f8.f4[0] += increment_f4;
                locInWindow_f8.f4[1] += increment_f4;
            }
            accum_f8.f4[0] += accum_f8.f4[1];
            accum += (accum_f8.f1[0] + accum_f8.f1[1] + accum_f8.f1[2] + accum_f8.f1[3]);   // perform small work of reducing float4 to float

            float x = locInWindow - inPos;
            for (; locInWindow < loc1; locInWindow++, x++)
            {
                float w = resample_hip_compute(x, windowScale, windowCenter, lookup_smem, lookupSize);
                accum += inBlockPtr[locInWindow] * w;
            }

            // Final store to dst
            dstPtr[dstIdx] = accum;
        // }
    }
    // copy input to output if dstLength is same as srcLength
    else
    {

        uint srcIdx = id_z * strides.x + outBlock;
        uint dstIdx = id_z * strides.y + outBlock;
        for (int outPos = outBlock; outPos < blockEnd; outPos++, dstIdx++, srcIdx++)
            dstPtr[dstIdx] = srcPtr[srcIdx];
    }
}

__global__ void resample_multi_channel_hip_tensor(float *srcPtr,
                                                  float *dstPtr,
                                                  uint2 strides,
                                                  int2 *srcDimsTensor,
                                                  int *dstDimsTensor,
                                                  RpptResamplingWindow *window)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    int srcLength = srcDimsTensor[id_z].x;
    int numChannels = srcDimsTensor[id_z].y;
    int dstLength = dstDimsTensor[id_z];
    int outBlock = id_x * hipBlockDim_x;
    int blockEnd = std::min(outBlock + static_cast<int>(hipBlockDim_x), dstLength);
    if (outBlock >= dstLength)
        return;

    double scale = static_cast<double>(srcLength) / dstLength;

    if (dstLength != srcLength)
    {
        extern __shared__ float lookup_smem[];

        // copy all values from window lookup table to shared memory lookup table
        for (int k = hipThreadIdx_x; k < window->lookupSize; k += hipBlockDim_x)
            lookup_smem[k] = window->lookup[k];
        __syncthreads();

        // extract the window scale, center and lookup size values from window
        float windowScale = window->scale;
        float windowCenter = window->center;
        int lookupSize = window->lookupSize;

        // compute block wise values required for processing
        double inBlockRaw = outBlock * scale;
        int inBlockRounded = static_cast<int>(inBlockRaw);
        float inPos = inBlockRaw - inBlockRounded;
        float fscale = scale;
        uint dstIdx = id_z * strides.y + outBlock * numChannels;
        float *inBlockPtr = srcPtr + id_z * strides.x + (inBlockRounded * numChannels);

        // process block size * channels (256 * channels) elements in single thread
        for (int outPos = outBlock; outPos < blockEnd; outPos++, inPos += fscale, dstIdx += numChannels)
        {
            int loc0, loc1;
            window->input_range(inPos, &loc0, &loc1);

            // check if computed loc0, loc1 values are beyond the input dimensions and update accordingly
            if (loc0 + inBlockRounded < 0)
                loc0 = -inBlockRounded;
            if (loc1 + inBlockRounded > srcLength)
                loc1 = srcLength - inBlockRounded;
            float locInWindow = loc0 - inPos;
            int2 offsetLocs_i2 = make_int2(loc0, loc1) * static_cast<int2>(numChannels);    // offsetted loc0, loc1 values for multi channel case

            float accum[RPPT_MAX_AUDIO_CHANNELS] = {0.0f};
            for (int offsetLoc = offsetLocs_i2.x; offsetLoc < offsetLocs_i2.y; offsetLoc += numChannels, locInWindow++)
            {
                float w = resample_hip_compute(locInWindow, windowScale, windowCenter, lookup_smem, lookupSize);
                for (int c = 0; c < numChannels; c++)
                    accum[c] += inBlockPtr[offsetLoc + c] * w;
            }

            // Final store to dst
            for (int c = 0; c < numChannels; c++)
                dstPtr[dstIdx + c] = accum[c];
        }
    }
    else
    {
        if (outBlock >= dstLength)
            return;

        uint srcIdx = id_z * strides.x + outBlock * numChannels;
        uint dstIdx = id_z * strides.y + outBlock * numChannels;
        for (int outPos = outBlock; outPos < blockEnd; outPos++, dstIdx += numChannels, srcIdx += numChannels)
        {
            for (int c = 0; c < numChannels; c++)
                dstPtr[dstIdx + c] = srcPtr[srcIdx + c];
        }
    }
}

// -------------------- Set 3 - resample kernels executor --------------------

RppStatus hip_exec_resample_tensor(Rpp32f *srcPtr,
                                   RpptDescPtr srcDescPtr,
                                   Rpp32f *dstPtr,
                                   RpptDescPtr dstDescPtr,
                                   Rpp32f *inRateTensor,
                                   Rpp32f *outRateTensor,
                                   Rpp32s *srcDimsTensor,
                                   RpptResamplingWindow &window,
                                   rpp::Handle& handle)
{
    Rpp32s globalThreads_x = dstDescPtr->strides.nStride;
    Rpp32s globalThreads_y = 1;
    Rpp32s globalThreads_z = dstDescPtr->n;
    Rpp32u tensorDims = srcDescPtr->numDims - 1; // exclude batchsize from input dims
    RppSize_t sharedMemorySizeInBytes = (window.lookupSize * sizeof(Rpp32f)); // shared memory size needed for resample kernel

    // using the input sampling rate, output sampling rate compute the output dims
    Rpp32s *dstDimsTensor = reinterpret_cast<Rpp32s *>(handle.GetInitHandle()->mem.mgpu.scratchBufferPinned.floatmem);
    // compute_output_dims(inRateTensor, outRateTensor, srcDimsTensor, dstDimsTensor, dstDescPtr->n);
    for (Rpp32s i = 0, j = 0; i < globalThreads_z; i++, j += 2)
        dstDimsTensor[i] = std::ceil(srcDimsTensor[j] * outRateTensor[i] / inRateTensor[i]);

    // For 1D audio tensors (channels = 1)
    if (tensorDims == 1)
    {
        hipLaunchKernelGGL(resample_single_channel_hip_tensor,
                           dim3(ceil((Rpp32f)globalThreads_x/256), ceil((Rpp32f)globalThreads_y/1), ceil((Rpp32f)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(256, 1, LOCAL_THREADS_Z_1DIM),
                           sharedMemorySizeInBytes,
                           handle.GetStream(),
                           srcPtr,
                           dstPtr,
                           make_uint2(srcDescPtr->strides.nStride, dstDescPtr->strides.nStride),
                           reinterpret_cast<int2 *>(srcDimsTensor),
                           dstDimsTensor,
                           &window);
    }
    // For 2D audio tensors (channels > 1)
    else if (tensorDims == 2)
    {
        hipLaunchKernelGGL(resample_multi_channel_hip_tensor,
                           dim3(ceil((Rpp32f)globalThreads_x/LOCAL_THREADS_X_1DIM), ceil((Rpp32f)globalThreads_y/LOCAL_THREADS_Y_1DIM), ceil((Rpp32f)globalThreads_z/LOCAL_THREADS_Z_1DIM)),
                           dim3(LOCAL_THREADS_X_1DIM, LOCAL_THREADS_Y_1DIM, LOCAL_THREADS_Z_1DIM),
                           sharedMemorySizeInBytes,
                           handle.GetStream(),
                           srcPtr,
                           dstPtr,
                           make_uint2(srcDescPtr->strides.nStride, dstDescPtr->strides.nStride),
                           reinterpret_cast<int2 *>(srcDimsTensor),
                           dstDimsTensor,
                           &window);
    }
    
    return RPP_SUCCESS;
}