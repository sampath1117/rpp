#include <hip/hip_runtime.h>
#include "rpp_hip_common.hpp"

__device__ void get_inverse_hip(d_float9 *mat, d_float9 *invMat)
{
    float det = mat->f1[0] * (mat->f1[4] * mat->f1[8] - mat->f1[7] * mat->f1[5]) - mat->f1[1] * (mat->f1[3] * mat->f1[8] - mat->f1[5] * mat->f1[6]) + mat->f1[2] * (mat->f1[3] * mat->f1[7] - mat->f1[4] * mat->f1[6]);
    if(det != 0)
    {
        float invDet = 1 / det;
        invMat->f1[0] = (mat->f1[4] * mat->f1[8] - mat->f1[7] * mat->f1[5]) * invDet;
        invMat->f1[1] = (mat->f1[2] * mat->f1[7] - mat->f1[1] * mat->f1[8]) * invDet;
        invMat->f1[2] = (mat->f1[1] * mat->f1[5] - mat->f1[2] * mat->f1[4]) * invDet;
        invMat->f1[3] = (mat->f1[5] * mat->f1[6] - mat->f1[3] * mat->f1[8]) * invDet;
        invMat->f1[4] = (mat->f1[0] * mat->f1[8] - mat->f1[2] * mat->f1[6]) * invDet;
        invMat->f1[5] = (mat->f1[3] * mat->f1[2] - mat->f1[0] * mat->f1[5]) * invDet;
        invMat->f1[6] = (mat->f1[3] * mat->f1[7] - mat->f1[6] * mat->f1[4]) * invDet;
        invMat->f1[7] = (mat->f1[6] * mat->f1[1] - mat->f1[0] * mat->f1[7]) * invDet;
        invMat->f1[8] = (mat->f1[0] * mat->f1[4] - mat->f1[3] * mat->f1[1]) * invDet;
    }
}

__global__ void compute_remap_tables(uint *rowRemapTable,
                                     uint *colRemapTable,
                                     d_float9 *cameraMatrixTensor,
                                     d_float9 *inverseMatrixTensor,
                                     d_float8 *distanceCoeffsTensor,
                                     d_float9 *newCameraMatrixTensor,
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

    d_float9 cameraMatrix = cameraMatrixTensor[id_z];
    d_float9 newCameraMatrix = newCameraMatrixTensor[id_z];
    d_float9 ir = inverseMatrixTensor[id_z];
    d_float8 distCoeffs = distanceCoeffsTensor[id_z];

    if(id_x == 0 && id_y == 0)
        get_inverse_hip(&newCameraMatrix, &ir);
    __syncthreads();

    float k1 = distCoeffs.f1[0], k2 = distCoeffs.f1[1];
    float p1 = distCoeffs.f1[2], p2 = distCoeffs.f1[3];
    float k3 = distCoeffs.f1[4], k4 = distCoeffs.f1[5], k5 = distCoeffs.f1[6], k6 = distCoeffs.f1[7];
    float u0 = cameraMatrix.f1[2],  v0 = cameraMatrix.f1[5];
    float fx = cameraMatrix.f1[0],  fy = cameraMatrix.f1[4];

    uint *rowRemapTableTemp = rowRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;
    uint *colRemapTableTemp = colRemapTable + id_z * remapTableStridesNH.x + id_y * remapTableStridesNH.y + id_x;

    float _x = id_y * ir.f1[1] + ir.f1[2] + id_x * ir.f1[0];
    float _y = id_y * ir.f1[4] + ir.f1[5] + id_x * ir.f1[3];
    float _w = id_y * ir.f1[7] + ir.f1[8] + id_x * ir.f1[6];

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
                                          Rpp32f *newCameraMatrix,
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
                           (d_float9 *)cameraMatrix,
                           (d_float9 *)inverseMatrix,
                           (d_float8 *)distanceCoeffs,
                           (d_float9 *)newCameraMatrix,
                           make_uint2(remapTableDescPtr->strides.nStride, remapTableDescPtr->strides.hStride),
                           roiTensorPtrSrc);
    }

    return RPP_SUCCESS;
}