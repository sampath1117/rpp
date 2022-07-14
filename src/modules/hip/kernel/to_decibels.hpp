#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void to_decibels_hip_compute(d_float8 *src_f8, d_float8 *dst_f8, float minRatio, float multiplier, float invReferenceMagnitude)
{
    dst_f8->f1[0] = multiplier * log10(fmaxf(minRatio, src_f8->f1[0] * invReferenceMagnitude));
    dst_f8->f1[1] = multiplier * log10(fmaxf(minRatio, src_f8->f1[1] * invReferenceMagnitude));
    dst_f8->f1[2] = multiplier * log10(fmaxf(minRatio, src_f8->f1[2] * invReferenceMagnitude));
    dst_f8->f1[3] = multiplier * log10(fmaxf(minRatio, src_f8->f1[3] * invReferenceMagnitude));
    dst_f8->f1[4] = multiplier * log10(fmaxf(minRatio, src_f8->f1[4] * invReferenceMagnitude));
    dst_f8->f1[5] = multiplier * log10(fmaxf(minRatio, src_f8->f1[5] * invReferenceMagnitude));
    dst_f8->f1[6] = multiplier * log10(fmaxf(minRatio, src_f8->f1[6] * invReferenceMagnitude));
    dst_f8->f1[7] = multiplier * log10(fmaxf(minRatio, src_f8->f1[7] * invReferenceMagnitude));
}

__global__ void to_decibels_tensor(float *srcPtr,
                                   uint2 srcStridesNH,
                                   float *dstPtr,
                                   uint *srcLengthTensor,
                                   float cutOffDB,
                                   float multiplier,
                                   float referenceMagnitude,
                                   float *maxValues)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 8;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if (id_x >= srcLengthTensor[id_z])
    {
        return;
    }

    uint loc = (id_z * srcStridesNH.x) + id_x;
    referenceMagnitude = (referenceMagnitude == 0.0) ? maxValues[id_z] : referenceMagnitude;
    // if(id_x == 0)
    // {
    //     printf("max for id_z %d = %f\n", id_z, referenceMagnitude);
    // }
    float invreferenceMagnitude = (1.0f / referenceMagnitude);
    float minRatio = pow(10, cutOffDB / multiplier);

    d_float8 src_f8, dst_f8;
    rpp_hip_load8_and_unpack_to_float8(srcPtr + loc, &src_f8);
    to_decibels_hip_compute(&src_f8, &dst_f8, minRatio, multiplier, invreferenceMagnitude);
    rpp_hip_pack_float8_and_store8(dstPtr + loc, &dst_f8);
}


__global__ void get_max(float *srcPtr,
                        uint *srcLength,
                        uint1 srcStride,
                        float *max,
                        int *mutex)
{
    int id_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
    int N = srcLength[id_z];

    if(id_x >= N)
        return;

    uint srcIdx = (id_z * srcStride.x) + id_x;
    int threadIdx = hipThreadIdx_x;
    int blockDimx = hipBlockDim_x;

    // Store block data in shared memory
    extern __shared__ float shm[];
    shm[threadIdx] = srcPtr[srcIdx];
    __syncthreads();

    // Do reduction
    for(int s = 1; s < hipBlockDim_x; s = s * 2)
    {
        int loc = 2 * s * threadIdx;
        if((loc + s) < N)
            shm[loc] = fmaxf(shm[loc], shm[loc + s]);

        __syncthreads();
    }

    // Get global maximum across blocks
    if(threadIdx == 0)
    {
		while(atomicCAS(mutex, 0, 1) != 0);  // lock
		max[id_z] = fmaxf(max[id_z], shm[0]);
		atomicExch(mutex, 0);  // unlock
	}
}

RppStatus hip_exec_to_decibels_tensor(Rpp32f *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f *dstPtr,
                                      Rpp32u *srcLengthTensor,
                                      Rpp32f cutOffDB,
                                      Rpp32f multiplier,
                                      Rpp32f referenceMagnitude,
                                      rpp::Handle& handle)
{
    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = 1;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    int globalThreads_y = 1;
    int globalThreads_z = handle.GetBatchSize();

    if(referenceMagnitude == 0.0)
    {
        int *mutex = nullptr;
        hipMalloc((void **)&mutex, sizeof(int));
        hipMemset(handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem, -std::numeric_limits<float>::max(), handle.GetBatchSize() * sizeof(float));
        hipMemset(mutex, 0, sizeof(int));

        globalThreads_x = srcDescPtr->strides.hStride;
        hipLaunchKernelGGL(get_max,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           LOCAL_THREADS_X,
                           handle.GetStream(),
                           srcPtr,
                           srcLengthTensor,
                           make_uint1(srcDescPtr->strides.nStride),
                           handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
                           mutex);
        hipDeviceSynchronize();
        hipFree(mutex);
    }

    globalThreads_x = (srcDescPtr->strides.hStride + 7) >> 3;
    hipLaunchKernelGGL(to_decibels_tensor,
                       dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       srcPtr,
                       make_uint2(srcDescPtr->strides.nStride, srcDescPtr->strides.hStride),
                       dstPtr,
                       srcLengthTensor,
                       cutOffDB,
                       multiplier,
                       referenceMagnitude,
                       handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem);

    return RPP_SUCCESS;
}
