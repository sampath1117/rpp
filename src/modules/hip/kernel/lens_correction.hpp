#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void get_inverse_hip(float *m, float *inv_m)
{
    float det = m[0] * (m[4] * m[8] - m[7] * m[5]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if(det != 0)
    {
        float invDet = 1 / det;
        inv_m[0] = (m[4] * m[8] - m[7] * m[5]) * invDet;
        inv_m[1] = (m[2] * m[7] - m[1] * m[8]) * invDet;
        inv_m[2] = (m[1] * m[5] - m[2] * m[4]) * invDet;
        inv_m[3] = (m[5] * m[6] - m[3] * m[8]) * invDet;
        inv_m[4] = (m[0] * m[8] - m[2] * m[6]) * invDet;
        inv_m[5] = (m[3] * m[2] - m[0] * m[5]) * invDet;
        inv_m[6] = (m[3] * m[7] - m[6] * m[4]) * invDet;
        inv_m[7] = (m[6] * m[1] - m[0] * m[7]) * invDet;
        inv_m[8] = (m[0] * m[4] - m[3] * m[1]) * invDet;
    }
}

__global__ void compute_remap_tables(uint *rowRemapTable,
                                     uint *colRemapTable,
                                     float *cameraMatrixTensor,
                                     float *inverseMatrixTensor,
                                     float *distanceCoeffsTensor,
                                     uint2 remapTableStridesNH,
                                     RpptROIPtr roiTensorPtrSrc)
{
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int id_z = hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;

    if ((id_y >= roiTensorPtrSrc[id_z].xywhROI.roiHeight) || (id_x >= roiTensorPtrSrc[id_z].xywhROI.roiWidth))
    {
        return;
    }

    int height = roiTensorPtrSrc[id_z].xywhROI.roiHeight;
    int width = roiTensorPtrSrc[id_z].xywhROI.roiWidth;

    float *cameraMatrix = cameraMatrixTensor + id_z * 9;
    float *distCoeffs = distanceCoeffsTensor + id_z * 14;
    float *ir = inverseMatrixTensor + id_z * 9;

    // if(id_x == 0 && id_y == 0)
    get_inverse_hip(cameraMatrix, ir);
    // __syncthreads();

    float k1 = distCoeffs[0], k2 = distCoeffs[1];
    float p1 = distCoeffs[2], p2 = distCoeffs[3];
    float k3 = distCoeffs[4], k4 = distCoeffs[5], k5 = distCoeffs[6], k6 = distCoeffs[7];
    float u0 = cameraMatrix[2],  v0 = cameraMatrix[5];
    float fx = cameraMatrix[0],  fy = cameraMatrix[4];

    uint *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;
    uint *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;

    float _x = id_y * ir[1] + ir[2] + id_x * ir[0];
    float _y = id_y * ir[4] + ir[5] + id_x * ir[3];
    float _w = id_y * ir[7] + ir[8] + id_x * ir[6];

    float w = 1./_w, x = _x * w, y = _y * w;
    float x2 = x * x, y2 = y * y;
    float r2 = x2 + y2, _2xy = 2 * x * y;
    float kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) *r2);
    float u = fx * (x * kr + p1 *_2xy + p2 * (r2 + 2 * x2)) + u0;
    float v = fy * (y * kr + p1 * (r2 + 2 * y2 ) + p2 *_2xy) + v0;
    int ui = floor(u);
    int vi = floor(v);
    *colRemapTableTemp = min(max(0, ui), width - 1);
    *rowRemapTableTemp = min(max(0, vi), height - 1);
}

// -------------------- Set 3 - Kernel Executors --------------------

RppStatus hip_exec_lens_correction_tensor(RpptDescPtr dstDescPtr,
                                          Rpp32u *rowRemapTable,
                                          Rpp32u *colRemapTable,
                                          RpptDescPtr remapTableDescPtr,
                                          Rpp32f *cameraMatrix,
                                          Rpp32f *distanceCoeffs,
                                          RpptROIPtr roiTensorPtrSrc,
                                          RpptRoiType roiType,
                                          rpp::Handle& handle)
{
    if (roiType == RpptRoiType::LTRB)
        hip_exec_roi_converison_ltrb_to_xywh(roiTensorPtrSrc, handle);

    int localThreads_x = LOCAL_THREADS_X;
    int localThreads_y = LOCAL_THREADS_Y;
    int localThreads_z = LOCAL_THREADS_Z;
    int globalThreads_x = dstDescPtr->w;
    int globalThreads_y = dstDescPtr->h;
    int globalThreads_z = handle.GetBatchSize();

    float *inverseMatrix = handle.GetInitHandle()->mem.mgpu.maskArr.floatmem;

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        hipLaunchKernelGGL(compute_remap_tables,
                           dim3(ceil((float)globalThreads_x/localThreads_x), ceil((float)globalThreads_y/localThreads_y), ceil((float)globalThreads_z/localThreads_z)),
                           dim3(localThreads_x, localThreads_y, localThreads_z),
                           0,
                           handle.GetStream(),
                           rowRemapTable,
                           colRemapTable,
                           cameraMatrix,
                           inverseMatrix,
                           distanceCoeffs,
                           make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}