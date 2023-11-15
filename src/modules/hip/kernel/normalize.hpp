#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"
#include "reduce_drop_dims.h"

int max_dims = 4;

/* norm_kernel.Run(ctx, out_view, in_view, mean_gpu, stddev_gpu,
                        scale_, shift_, epsilon); */

/**
 * @brief This variant is used when standard deviation is externally provided and needs to
 *        be regularized and inversed.
 *
 * The output elements are calculated as:
 * mul = 1 / sqrt(square(stddev[param_offset]) + epsilon)
 * (in[offset] - mean[param_offset]) * mul * scale + shift
 */
struct NormalizeInvStdDevNonScalar {
  float *out;
  float *in;
  int64_t size;
  const float *base;
  const float *scale;
  DropDims dd;

    __host__ __device__ __forceinline__ void apply(int64_t offset, float epsilon, float global_scale, float global_shift)
    {
        int64_t param_offset = dd.reindex(offset);
        float mean = base[param_offset];
        float stddev = scale[param_offset];
        float x = fmaf(stddev, stddev, epsilon);
        float mul = x ? rsqrt(x) * global_scale : 0;
        out[offset] = fmaf(in[offset] - mean, mul, global_shift);
    }
};

template <typename NormalizeParams>
__global__ void NormalizeInvStdDevKernel(const NormalizeParams *sample_params,
                                         float epsilon, float scale, float shift) {
  auto params = sample_params[hipBlockIdx_y];
  int64_t start_ofs = static_cast<int64_t>(hipBlockIdx_x) * hipBlockDim_x + hipThreadIdx_x;
  int64_t grid_stride = static_cast<int64_t>(hipGridDim_x) * hipBlockDim_x;
  for (int64_t ofs = start_ofs; ofs < params.size; ofs += grid_stride) {
    params.apply(ofs, epsilon, scale, shift);
  }
}

int64_t volume(Rpp32u *roi, Rpp32u input_dims)
{
    int64_t size = 1;
    for(int i = 0; i < input_dims; i++)
        size *= roi[i];
    return size;
}

void FillDescs(NormalizeInvStdDevNonScalar *descs,
               float *out,
               float *in,
               float *base,
               float *scale,
               Rpp32u axis_mask,
               Rpp32u *roiTensor,
               RpptGenericDescPtr srcGenericDescPtr,
               RpptGenericDescPtr dstGenericDescPtr) {
    int num_samples_ = srcGenericDescPtr->dims[0];
    int base_idx_delta = num_samples_ == 1 ? 0 : 1;
    int scale_idx_delta = num_samples_ == 1 ? 0 : 1;
    Rpp32u input_dims = srcGenericDescPtr->numDims - 1;
    Rpp32u param_stride;

    // loop over all samples in a batch
    for (int i = 0, b = 0, s = 0; i < num_samples_;
         i++, b += base_idx_delta, s += scale_idx_delta) {
        Rpp32f *outTemp = out + i * dstGenericDescPtr->strides[0];
        Rpp32f *inTemp = in + i * srcGenericDescPtr->strides[0];
        Rpp32f *baseTemp = base + i * param_stride;
        Rpp32f *scaleTemp = scale + i * param_stride;
        Rpp32u *length = roiTensor + i * input_dims * 2 + input_dims;
        auto &desc = descs[i];
        desc.out = outTemp;
        desc.in = inTemp;
        desc.scale = scaleTemp;
        desc.base = baseTemp;
        desc.size = volume(length, input_dims);
        std::cout << "volume: " << desc.size << std::endl;
        desc.dd = DropDims(reinterpret_cast<int64_t *>(length), static_cast<uint64_t>(axis_mask), input_dims);
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
                                    Rpp32f global_scale,
                                    Rpp32f shift,
                                    Rpp32u *roiTensor,
                                    rpp::Handle& handle)
{
    Rpp32u batchSize = srcGenericDescPtr->dims[0];

    // fill description pointer needed for kernel processing
    NormalizeInvStdDevNonScalar *cpu_descs;
    hipHostMalloc(&cpu_descs, batchSize * sizeof(NormalizeInvStdDevNonScalar));
    FillDescs(cpu_descs, dstPtr, srcPtr,
              meanTensor, stdDevTensor, axisMask,
              roiTensor, srcGenericDescPtr, dstGenericDescPtr);

    // set block and grid values
    int localThreads_x = std::min(256, (int)srcGenericDescPtr->strides[0]);
    int localThreads_y = 1;
    int localThreads_z = 1;

    int globalThreads_x = ceil((float)srcGenericDescPtr->strides[0] / localThreads_x);
    int globalThreads_y = 1;
    int globalThreads_z = 1;

    // launch kernel
    Rpp32f epsilon;
    hipLaunchKernelGGL(NormalizeInvStdDevKernel,
                       dim3(globalThreads_x, globalThreads_y, globalThreads_z),
                       dim3(localThreads_x, localThreads_y, localThreads_z),
                       0,
                       handle.GetStream(),
                       cpu_descs,
                       epsilon,
                       global_scale,
                       shift);

    hipHostFree(cpu_descs);
    return RPP_SUCCESS;
}