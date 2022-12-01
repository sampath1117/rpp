#include "rppdefs.h"
#include "rpp_cpu_simd.hpp"
#include "rpp_cpu_common.hpp"

#define PACK 8

float accumalate_ps(__m256 src) {
    __m256 src_add = _mm256_add_ps(src, _mm256_permute2f128_ps(src, src, 1));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(1, 0, 3, 2)));
    src_add = _mm256_add_ps(src_add, _mm256_shuffle_ps(src_add, src_add, _MM_SHUFFLE(2, 3, 0, 1)));
    float* addResult = (float*)&src_add;
    return addResult[0];
}

__m256 mask_mov_ps(__m256 src, unsigned short msk, __m256 set1) {
    __m256i bitsArr = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    //Checks if specific is set and if so makes all bits high for corresponding element in bitsSet
    __m256i bitsSet = _mm256_and_si256(bitsArr, _mm256_set1_epi32(msk));
    bitsSet = _mm256_cmpeq_epi32(bitsSet, bitsArr);
    //Stores the complement of the elements in bitsSet
    __m256i bitsUnset = _mm256_xor_si256(bitsSet, _mm256_set1_epi32(0xFFFFFFFF));
    //setResult stores the values of set1 based on the bitsSet
    __m256 setResult = _mm256_and_ps(set1, _mm256_castsi256_ps(bitsSet));
    //Stores the elements of src wherever bits are not set in the mask
    __m256 unsetResult = _mm256_and_ps(src, _mm256_castsi256_ps(bitsUnset));
    //Computes the final results based on the previous results
    __m256 finalResult = _mm256_add_ps(setResult, unsetResult);
    return finalResult;
}

__m256i mask_mov_epi32(__m256i src, unsigned short msk, __m256i set1) {
    __m256i bitsArr = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);
    //Checks if specific is set and if so makes all bits high for corresponding element in bitsSet
    __m256i bitsSet = _mm256_and_si256(bitsArr, _mm256_set1_epi32(msk));
    bitsSet = _mm256_cmpeq_epi32(bitsSet, bitsArr);
    //Stores the complement of the elements in bitsSet
    __m256i bitsUnset = _mm256_xor_si256(bitsSet, _mm256_set1_epi32(0xFFFFFFFF));
    //setResult stores the values of set1 based on the bitsSet
    __m256i setResult = _mm256_and_si256(set1, (bitsSet));
    //Stores the elements of src wherever bits are not set in the mask
    __m256i unsetResult = _mm256_and_si256(src, (bitsUnset));
    //Computes the final results based on the previous results
    __m256i finalResult = _mm256_add_epi32(setResult, unsetResult);
    return finalResult;
}

__m256i maskGenerate_epi32(int32_t value, int rem) {
    __m256i k1_n = _mm256_setzero_si256();
    __m256i k2_n = _mm256_set1_epi32((int32_t) value);
    unsigned char m1 = pow(2,rem) - 1;
    __m256i k_n = mask_mov_epi32(k1_n, m1, k2_n);
    return k_n;
}

__m256 maskGenerate_ps(float value, int rem) {
    __m256 k1_n = _mm256_setzero_ps();
    __m256 k2_n = _mm256_set1_ps(value);
    unsigned char m1 = pow(2,rem) - 1;
    __m256 k_n = mask_mov_ps(k1_n, m1, k2_n);
    return k_n;
}


void compute_2D_mean(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    int rem = dims[1]%PACK;
    int v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;
    __m256 pack_n = _mm256_set1_ps(PACK);
    __m256 stride_n = _mm256_set1_ps(stride[0]);
    for(Rpp32u i = 0; i < dims[0]; i++) {
        meanPtr[i] = 0;
        __m256 j_n = _mm256_set_ps(7,6,5,4,3,2,1,0);
        __m256i k_n = _mm256_set1_epi32((int32_t)2147483648);
        for(Rpp32u j = 0; j < v_n; j++) {
            if ((j == (v_n - 1)) && (rem != 0)) {
                k_n = maskGenerate_epi32((int32_t)2147483648, rem);
            }
            //meanPtr[i] += (*(srcPtrTemp + j * stride[0]));
            __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
            __m256 meanPtr_n = _mm256_mask_i32gather_ps( _mm256_setzero_ps(), srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), _mm256_castsi256_ps(k_n), 4);
            meanPtr[i] += accumalate_ps(meanPtr_n);
            j_n = _mm256_add_ps(j_n, pack_n);
        }
        srcPtrTemp += stride[1];
        meanPtr[i] = meanPtr[i] / dims[1];
    }
}

void compute_2D_mean_axis1(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride) {
    // Set total length and calculate rem
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u rem = dims[1]%PACK;
    Rpp32u v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    __m256 pack_n = _mm256_set1_ps(PACK);
    __m256 stride_n = _mm256_set1_ps(stride[0]);

    // Outer loop with channels
    for(Rpp32u i = 0; i < dims[0]; i++) {
        meanPtr[i] = 0;

        __m256 j_n = _mm256_set_ps(7,6,5,4,3,2,1,0);
        __m256 meanPtr_n = _mm256_setzero_ps();

        // Inner loop with length
        for(Rpp32u j = 0; j < (v_n - 1); j++) {
            //meanPtr[i] += (*(srcPtrTemp + j * stride[0]));
            __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
            __m256 srcPtrTemp_n = _mm256_i32gather_ps(srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), 4);
            meanPtr_n = _mm256_add_ps(srcPtrTemp_n, meanPtr_n);
            j_n = _mm256_add_ps(j_n, pack_n);
        }

        __m256i k_n;
        // Rem condition
        if (rem != 0) {
            k_n = maskGenerate_epi32((int32_t)2147483648, rem);
        } else {
            k_n = _mm256_set1_epi32((int32_t)2147483648);
        }

        __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
        __m256 srcPtrTemp_n = _mm256_mask_i32gather_ps( _mm256_setzero_ps(), srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), _mm256_castsi256_ps(k_n), 4);
        meanPtr_n = _mm256_add_ps(srcPtrTemp_n, meanPtr_n);

        meanPtr[i] = accumalate_ps(meanPtr_n);
        meanPtr[i] = meanPtr[i] / dims[1];

        srcPtrTemp += stride[1];
    }
}

void compute_2D_mean_axis2_4samples(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *meanPtrTemp = meanPtr;

    int val_stride = PACK/2;

    Rpp32u rem_sampleLength = dims[0]%val_stride;
    Rpp32u samplelength_n = (!rem_sampleLength) ? dims[0]/val_stride: (dims[0]/val_stride)+1;

    Rpp32u rem_channels = dims[1]%val_stride;
    Rpp32u channel_n = (!rem_channels) ? dims[1]/val_stride: (dims[1]/val_stride)+1;

    __m128 div_channels_n = _mm_set1_ps(1.0/dims[1]);
    if (dims[1] <= 4) {
        if (rem_sampleLength != 0) {
            samplelength_n -= 1;
        }
        // Outer loop source length
        for(Rpp32u i = 0; i < samplelength_n; i++) {
            if (dims[1] == 1) {
                __m256 srcPtrTemp_n = _mm256_loadu_ps(srcPtrTemp);
                _mm256_storeu_ps(meanPtrTemp, srcPtrTemp_n);
                srcPtrTemp+=PACK;
                meanPtrTemp+=PACK;
                continue;
            } else if (dims[1] == 2) {
                __m128 r0,r1,r2,r3;
                __m128 meanptr_n = _mm_setzero_ps();
                __m128 srcPtrTemp_n_0 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp +=val_stride;
                __m128 srcPtrTemp_n_1 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp +=val_stride;
                r0 = _mm_shuffle_ps(srcPtrTemp_n_0, srcPtrTemp_n_1, 136);
                r1 = _mm_shuffle_ps(srcPtrTemp_n_0, srcPtrTemp_n_1, 221);
                meanptr_n = _mm_add_ps(r0,r1);
                meanptr_n = _mm_mul_ps(meanptr_n, div_channels_n);
                _mm_storeu_ps(meanPtrTemp, meanptr_n);
                meanPtrTemp += val_stride;
                continue;
            } else if (dims[1] == 3) {
                __m128 r0,r1,r2,r3;
                __m128 meanptr_n = _mm_setzero_ps();
                r0 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                r1 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                r2 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                r3 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                _MM_TRANSPOSE4_PS(r0,r1,r2,r3);
                meanptr_n = _mm_add_ps(r0,r1);
                meanptr_n = _mm_add_ps(r2,meanptr_n);
                meanptr_n = _mm_mul_ps(meanptr_n, div_channels_n);
                _mm_storeu_ps(meanPtrTemp, meanptr_n);
                meanPtrTemp += val_stride;
                continue;
            } else if (dims[1] == 4) {
                __m128 r0,r1,r2,r3;
                __m128 meanptr_n = _mm_setzero_ps();
                r0 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                r1 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                r2 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                r3 = _mm_loadu_ps(srcPtrTemp);
                srcPtrTemp += dims[1];
                _MM_TRANSPOSE4_PS(r0,r1,r2,r3);
                meanptr_n = _mm_add_ps(r0,r1);
                meanptr_n = _mm_add_ps(r2,meanptr_n);
                meanptr_n = _mm_add_ps(r3,meanptr_n);
                meanptr_n = _mm_mul_ps(meanptr_n, div_channels_n);
                _mm_storeu_ps(meanPtrTemp, meanptr_n);
                meanPtrTemp += val_stride;
                continue;
            }
        }

        // Rem samples
        if (rem_sampleLength != 0) {
            Rpp32u doneSamples = samplelength_n*val_stride;
            for (Rpp32u i = 0; i < rem_sampleLength; i++) {
               for (Rpp32u j = 0; j < dims[1]; j++) {
                 meanPtr[doneSamples] += *srcPtrTemp;
                 srcPtrTemp += 1;
               }
               meanPtr[doneSamples] /= dims[1];
               doneSamples += 1;
            }
        }
    } else  {
        if (rem_sampleLength != 0) {
            samplelength_n -= 1;
        }
        if (rem_channels != 0) {
            channel_n -= 1;
        }

        // Outer loop source length
        for(Rpp32u i = 0; i < samplelength_n; i++) {
            Rpp32f* srcPtrTemp_0 = srcPtrTemp;
            Rpp32f* srcPtrTemp_1 = srcPtrTemp + (dims[1]);
            Rpp32f* srcPtrTemp_2 = srcPtrTemp + (dims[1] * 2);
            Rpp32f* srcPtrTemp_3 = srcPtrTemp + (dims[1] * 3);
            __m128 meanptr_n = _mm_setzero_ps();
            for(Rpp32u j = 0; j < channel_n; j++) {
                __m128 r0,r1,r2,r3;
                r0 = _mm_loadu_ps(srcPtrTemp_0);
                srcPtrTemp_0 += val_stride;
                r1 = _mm_loadu_ps(srcPtrTemp_1);
                srcPtrTemp_1 += val_stride;
                r2 = _mm_loadu_ps(srcPtrTemp_2);
                srcPtrTemp_2 += val_stride;
                r3 = _mm_loadu_ps(srcPtrTemp_3);
                srcPtrTemp_3 += val_stride;
                _MM_TRANSPOSE4_PS(r0,r1,r2,r3);
                meanptr_n = _mm_add_ps(r0,meanptr_n);
                meanptr_n = _mm_add_ps(r1,meanptr_n);
                meanptr_n = _mm_add_ps(r2,meanptr_n);
                meanptr_n = _mm_add_ps(r3,meanptr_n);
            }
            if (rem_channels != 0) {
                __m128 r0,r1,r2,r3;
                r0 = _mm_loadu_ps(srcPtrTemp_0);
                r1 = _mm_loadu_ps(srcPtrTemp_1);
                r2 = _mm_loadu_ps(srcPtrTemp_2);
                r3 = _mm_loadu_ps(srcPtrTemp_3);
                _MM_TRANSPOSE4_PS(r0,r1,r2,r3);
                if (rem_channels == 1) {
                    meanptr_n = _mm_add_ps(r0,meanptr_n);
                } else if (rem_channels == 2) {
                    meanptr_n = _mm_add_ps(r0,meanptr_n);
                    meanptr_n = _mm_add_ps(r1,meanptr_n);
                } else if (rem_channels == 3) {
                    meanptr_n = _mm_add_ps(r0,meanptr_n);
                    meanptr_n = _mm_add_ps(r1,meanptr_n);
                    meanptr_n = _mm_add_ps(r2,meanptr_n);
                }
            }
            meanptr_n = _mm_mul_ps(meanptr_n, div_channels_n);
            _mm_storeu_ps(meanPtrTemp, meanptr_n);
            meanPtrTemp += val_stride;
            srcPtrTemp += (val_stride * dims[1]);
        }
        // Rem samples
        if (rem_sampleLength != 0) {
            Rpp32u doneSamples = samplelength_n*val_stride;
            for (Rpp32u i = 0; i < rem_sampleLength; i++) {
               for (Rpp32u j = 0; j < dims[1]; j++) {
                 meanPtr[doneSamples] += *srcPtrTemp;
                 srcPtrTemp += 1;
               }
               meanPtr[doneSamples] /= dims[1];
               doneSamples += 1;
            }
        }
    }
}

void compute_2D_mean_axis2(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u rem = dims[1]%PACK;
    Rpp32u v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    // Outer loop source length
    for(Rpp32u i = 0; i < dims[0]; i++) {
        meanPtr[i] = 0;

        __m256 meanPtr_n  = _mm256_setzero_ps();

        // Inner loop channel
        for(Rpp32u j = 0; j < (v_n - 1); j++) {
            //meanPtr[i] += (*(srcPtrTemp + j * stride[0]));
            __m256 srcPtrTemp_n = _mm256_loadu_ps(srcPtrTemp);
            meanPtr_n = _mm256_add_ps(srcPtrTemp_n, meanPtr_n);
            srcPtrTemp +=PACK;
        }

        __m256i k_n;
        // Rem condition
        if (rem != 0) {
            k_n = maskGenerate_epi32((int32_t)2147483648, rem);
        } else {
            k_n = _mm256_set1_epi32((int32_t)2147483648);
        }

        __m256 srcPtrTemp_n = _mm256_maskload_ps(srcPtrTemp, k_n);
        meanPtr_n = _mm256_add_ps(srcPtrTemp_n, meanPtr_n);

        srcPtrTemp += rem;

        meanPtr[i] += accumalate_ps(meanPtr_n);
        meanPtr[i] = meanPtr[i] / dims[1];
    }
}

void compute_2D_mean_axis3(Rpp32f *srcPtr, Rpp32f *meanPtr,  Rpp32u *dims, Rpp32u *stride) {
    // Set total length and calculate rem
    Rpp32f *srcPtrTemp = srcPtr;

    Rpp32u totalLength = dims[0]*dims[1];
    Rpp32u rem = (totalLength)%PACK;
    Rpp32u v_n = (!rem) ? totalLength/PACK: (totalLength/PACK)+1;
    meanPtr[0] = 0;

    __m256 meanPtr_n = _mm256_setzero_ps();

    // Total length loop
    for(Rpp32u j = 0; j < (v_n - 1); j++) {
        //meanPtr[i] += (*(srcPtrTemp + j * stride[0]));
        meanPtr_n = _mm256_add_ps(_mm256_loadu_ps(srcPtrTemp), meanPtr_n);
        srcPtrTemp += PACK;
    }

    __m256i k_n;

    // Rem condition
    if (rem != 0) {
        k_n = maskGenerate_epi32((int32_t)2147483648, rem);
    } else {
        k_n = _mm256_set1_epi32((int32_t)2147483648);
    }

    meanPtr_n = _mm256_add_ps(_mm256_maskload_ps(srcPtrTemp, k_n), meanPtr_n);
    meanPtr[0] = accumalate_ps(meanPtr_n);
    meanPtr[0] = meanPtr[0] / totalLength;
}

void compute_2D_inv_std_dev(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    int rem = dims[1]%PACK;
    int v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;
    __m256 pack_n = _mm256_set1_ps(PACK);
    __m256 stride_n = _mm256_set1_ps(stride[0]);
    for(Rpp32u i = 0; i < dims[0]; i++) {
        stdDevPtr[i] = 0;
        __m256 j_n = _mm256_set_ps(7,6,5,4,3,2,1,0);
        __m256 meanptr_n = _mm256_set1_ps(meanPtr[i]);
        __m256i k_n = _mm256_set1_epi32((int32_t)2147483648);
        for(Rpp32u j = 0; j < v_n; j++) {
            if ((j == (v_n - 1)) && (rem != 0)) {
              k_n = maskGenerate_epi32((int32_t)2147483648, rem);
              meanptr_n = maskGenerate_ps(meanPtr[i], rem);
            }
            //Rpp32f diff = (*(srcPtrTemp + j * stride[0]) - meanPtr[i]);
            //stdDevPtr[i] += (diff * diff);
            __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
            __m256 diff_n = _mm256_sub_ps(_mm256_mask_i32gather_ps(_mm256_setzero_ps(), srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), _mm256_castsi256_ps(k_n), 4), meanptr_n);
            stdDevPtr[i] += accumalate_ps(_mm256_mul_ps(diff_n, diff_n));
            j_n = _mm256_add_ps(j_n, pack_n);
        }
        srcPtrTemp += stride[1];
        stdDevPtr[i] = stdDevPtr[i] / dims[1];
        stdDevPtr[i] = (!stdDevPtr[i]) ? 0.0f : 1.0f / sqrt(stdDevPtr[i]);
    }
}

void compute_2D_inv_std_dev_axis1(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u rem = dims[1]%PACK;
    Rpp32u v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    __m256 pack_n = _mm256_set1_ps(PACK);
    __m256 stride_n = _mm256_set1_ps(stride[0]);

    // Outer loop channels
    for(Rpp32u i = 0; i < dims[0]; i++) {
        stdDevPtr[i] = 0;

        __m256 j_n = _mm256_set_ps(7,6,5,4,3,2,1,0);
        __m256 meanptr_n = _mm256_set1_ps(meanPtr[i]);
        __m256 stdDevPtr_n = _mm256_setzero_ps();


        // Inner loop length
        for(Rpp32u j = 0; j < (v_n - 1); j++) {
            //Rpp32f diff = (*(srcPtrTemp + j * stride[0]) - meanPtr[i]);
            //stdDevPtr[i] += (diff * diff);
            __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
            __m256 diff_n = _mm256_sub_ps(_mm256_i32gather_ps(srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), 4), meanptr_n);
            stdDevPtr_n = _mm256_add_ps(stdDevPtr_n, _mm256_mul_ps(diff_n, diff_n));
            j_n = _mm256_add_ps(j_n, pack_n);
        }

        __m256i k_n;
        // Rem condition
        if ((rem != 0)) {
            k_n = maskGenerate_epi32((int32_t)2147483648, rem);
            meanptr_n = maskGenerate_ps(meanPtr[i], rem);
        } else {
            k_n = _mm256_set1_epi32((int32_t)2147483648);
        }

        __m256 stride_j_n = _mm256_mul_ps(j_n, stride_n);
        __m256 diff_n = _mm256_sub_ps(_mm256_mask_i32gather_ps(_mm256_setzero_ps(), srcPtrTemp, _mm256_cvtps_epi32(stride_j_n), _mm256_castsi256_ps(k_n), 4), meanptr_n);
        stdDevPtr_n = _mm256_add_ps(stdDevPtr_n, _mm256_mul_ps(diff_n, diff_n));

        stdDevPtr[i] = accumalate_ps(stdDevPtr_n);

        stdDevPtr[i] = stdDevPtr[i] / dims[1];
        stdDevPtr[i] = (!stdDevPtr[i]) ? 0.0f : 1.0f / sqrt(stdDevPtr[i]);

        srcPtrTemp += stride[1];
    }
}

void compute_2D_inv_std_dev_axis2(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u rem = dims[1]%PACK;
    Rpp32u v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    // Outer loop source length
    for(Rpp32u i = 0; i < dims[0]; i++) {
        stdDevPtr[i] = 0;
        __m256 meanptr_n = _mm256_set1_ps(meanPtr[i]);
        __m256 stdDevPtr_n = _mm256_setzero_ps();

        // Inner loop channels
        for(Rpp32u j = 0; j < (v_n-1); j++) {
            //Rpp32f diff = (*(srcPtrTemp + j * stride[0]) - meanPtr[i]);
            //stdDevPtr[i] += (diff * diff);
            __m256 diff_n = _mm256_sub_ps(_mm256_loadu_ps(srcPtrTemp), meanptr_n);
            stdDevPtr_n = _mm256_add_ps(stdDevPtr_n, _mm256_mul_ps(diff_n, diff_n));
            srcPtrTemp += PACK;
        }

        __m256i k_n;
        // Rem condition
        if (rem != 0) {
            k_n = maskGenerate_epi32((int32_t)2147483648, rem);
            meanptr_n = maskGenerate_ps(meanPtr[i], rem);
        } else {
            k_n = _mm256_set1_epi32((int32_t)2147483648);
        }

        __m256 diff_n = _mm256_sub_ps(_mm256_maskload_ps(srcPtrTemp, k_n), meanptr_n);
        stdDevPtr_n = _mm256_add_ps(stdDevPtr_n, _mm256_mul_ps(diff_n, diff_n));

        srcPtrTemp += rem;

        stdDevPtr[i] += accumalate_ps(stdDevPtr_n);
        stdDevPtr[i] = stdDevPtr[i] / dims[1];
        stdDevPtr[i] = (!stdDevPtr[i]) ? 0.0f : 1.0f / sqrt(stdDevPtr[i]);
    }
}

void compute_2D_inv_std_dev_axis3(Rpp32f *srcPtr, Rpp32f *meanPtr, Rpp32f *stdDevPtr, Rpp32u *dims, Rpp32u *stride) {
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32u totalLength = dims[0] * dims[1];
    Rpp32u rem = totalLength%PACK;
    Rpp32u v_n = (!rem) ? totalLength/PACK: (totalLength/PACK)+1;

    __m256 meanptr_n = _mm256_set1_ps(meanPtr[0]);
    __m256 stdDevPtr_n = _mm256_setzero_ps();

    stdDevPtr[0] = 0;

    for(Rpp32u j = 0; j < (v_n - 1); j++) {
        //Rpp32f diff = (*(srcPtrTemp + j * stride[0]) - meanPtr[i]);
        //stdDevPtr[i] += (diff * diff);
        __m256 diff_n = _mm256_sub_ps(_mm256_loadu_ps(srcPtrTemp) , meanptr_n);
        stdDevPtr_n = _mm256_add_ps(_mm256_mul_ps(diff_n, diff_n), stdDevPtr_n);
        srcPtrTemp += PACK;
    }

    __m256i k_n;
    // Rem condition
    if (rem != 0) {
        k_n = maskGenerate_epi32((int32_t)2147483648, rem);
        meanptr_n = maskGenerate_ps(meanPtr[0], rem);
    } else {
        k_n = _mm256_set1_epi32((int32_t)2147483648);
    }

    __m256 diff_n = _mm256_sub_ps(_mm256_maskload_ps(srcPtrTemp, k_n), meanptr_n);
    stdDevPtr_n = _mm256_add_ps(_mm256_mul_ps(diff_n, diff_n), stdDevPtr_n);

    stdDevPtr[0] = accumalate_ps(stdDevPtr_n);
    stdDevPtr[0] = stdDevPtr[0] / totalLength;
    stdDevPtr[0] = (!stdDevPtr[0]) ? 0.0f : 1.0f / sqrt(stdDevPtr[0]);
}

void normalize_2D_tensor_cpu(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32s paramIdx = 0;
    for(Rpp32u i = 0; i < dims[0]; i++) {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;
        for(Rpp32u j = 0; j < dims[1]; j++) {
            *dstPtrTempRow++ = (*srcPtrTempRow++ - meanPtr[paramIdx]) * invStdDevPtr[paramIdx] + shift;
            paramIdx += paramStride[0];
        }
        paramIdx = (!paramStride[1]) ? 0 : paramIdx + paramStride[1];
        srcPtrTemp += srcDescPtr->strides.wStride;
        dstPtrTemp += dstDescPtr->strides.wStride;
    }
}

void normalize_2D_tensor_allaxis(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;

    int rem = dims[1]%PACK;
    int v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    __m256 meanptr_n, invStdDevPtr_n;

    for(Rpp32u i = 0; i < dims[0]; i++) {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;

        // Set all params for i loop
        // axis 3
        if (paramStride[0] == 0 && paramStride[1] == 0) {
            meanptr_n = _mm256_set1_ps(meanPtr[0]);
            invStdDevPtr_n = _mm256_set1_ps(invStdDevPtr[0]);
        }
        // axis 2
        if (paramStride[0] == 0 && paramStride[1] == 1) {
            meanptr_n = _mm256_set1_ps(meanPtr[i]);
            invStdDevPtr_n = _mm256_set1_ps(invStdDevPtr[i]);
        }

        __m256i k_n = _mm256_set1_epi32((int32_t)2147483648);
        __m256 shift_n = _mm256_set1_ps(shift);

        for(Rpp32u j = 0; j < v_n; j++) {
            __m256 srcPtrTempRow_n;

            if ((j == (v_n - 1)) && (rem != 0)) {
                k_n = maskGenerate_epi32((int32_t)2147483648, rem);
                shift_n = maskGenerate_ps(shift, rem);
                // axis 2
                if (paramStride[0] == 0 && paramStride[1] == 1) {
                    meanptr_n = maskGenerate_ps(meanPtr[i], rem);
                    invStdDevPtr_n = maskGenerate_ps(invStdDevPtr[i], rem);
                } else if (paramStride[0] == 1 && paramStride[1] == 0) { // axis 1
                    meanptr_n = _mm256_maskload_ps(meanPtr+(j*PACK), k_n);
                    invStdDevPtr_n = _mm256_maskload_ps(invStdDevPtr+(j*PACK), k_n);
                }
                srcPtrTempRow_n = _mm256_maskload_ps(srcPtrTempRow, k_n);
            } else {
                // axis 1
                if (paramStride[0] == 1 && paramStride[1] == 0) {
                    meanptr_n = _mm256_load_ps(meanPtr+(j*PACK));
                    invStdDevPtr_n = _mm256_load_ps(invStdDevPtr+(j*PACK));
                }
                srcPtrTempRow_n = _mm256_load_ps(srcPtrTempRow);
            }

            __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
            _mm256_maskstore_ps(dstPtrTempRow, k_n, dstPtrTempRow_n);
            srcPtrTempRow += PACK;
            dstPtrTempRow += PACK;
        }
        srcPtrTemp += srcDescPtr->strides.wStride;
        dstPtrTemp += dstDescPtr->strides.wStride;
    }
}

void normalize_2D_tensor_axis1(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32u rem = dims[1]%PACK;
    Rpp32u v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    __m256 meanptr_n, invStdDevPtr_n;

    for(Rpp32u i = 0; i < dims[0]; i++) {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;

        // Set all params for i loop
        __m256 shift_n = _mm256_set1_ps(shift);
        __m256 srcPtrTempRow_n;

        Rpp32u j = 0;
        for(; j < (v_n - 1); j++) {
            meanptr_n = _mm256_loadu_ps(meanPtr+(j*PACK));
            invStdDevPtr_n = _mm256_loadu_ps(invStdDevPtr+(j*PACK));
            srcPtrTempRow_n = _mm256_loadu_ps(srcPtrTempRow);

            __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
            _mm256_storeu_ps(dstPtrTempRow, dstPtrTempRow_n);
            srcPtrTempRow += PACK;
            dstPtrTempRow += PACK;
        }

        // Rem condition
        __m256i k_n;
        if (rem != 0) {
            k_n = maskGenerate_epi32((int32_t)2147483648, rem);
            shift_n = maskGenerate_ps(shift, rem);
        } else {
            k_n = _mm256_set1_epi32((int32_t)2147483648);
        }

        meanptr_n = _mm256_maskload_ps(meanPtr+(j*PACK), k_n);
        invStdDevPtr_n = _mm256_maskload_ps(invStdDevPtr+(j*PACK), k_n);
        srcPtrTempRow_n = _mm256_maskload_ps(srcPtrTempRow, k_n);

        __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
        _mm256_maskstore_ps(dstPtrTempRow, k_n, dstPtrTempRow_n);

        srcPtrTemp += srcDescPtr->strides.wStride;
        dstPtrTemp += dstDescPtr->strides.wStride;
    }
}

void normalize_2D_tensor_axis2(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;

    Rpp32u rem = dims[1]%PACK;
    Rpp32u v_n = (!rem) ? dims[1]/PACK: (dims[1]/PACK)+1;

    __m256 meanptr_n, invStdDevPtr_n;

    for(Rpp32u i = 0; i < dims[0]; i++) {
        Rpp32f *srcPtrTempRow = srcPtrTemp;
        Rpp32f *dstPtrTempRow = dstPtrTemp;

        // Set all params for i loop
        meanptr_n = _mm256_set1_ps(meanPtr[i]);
        invStdDevPtr_n = _mm256_set1_ps(invStdDevPtr[i]);
        __m256 shift_n = _mm256_set1_ps(shift);
        __m256 srcPtrTempRow_n;

        Rpp32u j = 0;
        for(; j < (v_n - 1); j++) {
            srcPtrTempRow_n = _mm256_loadu_ps(srcPtrTempRow);
            __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
            _mm256_storeu_ps(dstPtrTempRow, dstPtrTempRow_n);
            srcPtrTempRow += PACK;
            dstPtrTempRow += PACK;
        }

        // Rem condition
        __m256i k_n;
        if (rem != 0) {
            k_n = maskGenerate_epi32((int32_t)2147483648, rem);
            shift_n = maskGenerate_ps(shift, rem);
            meanptr_n = maskGenerate_ps(meanPtr[i], rem);
            invStdDevPtr_n = maskGenerate_ps(invStdDevPtr[i], rem);
        } else {
            k_n = _mm256_set1_epi32((int32_t)2147483648);
        }

        srcPtrTempRow_n = _mm256_maskload_ps(srcPtrTempRow, k_n);

        __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
        _mm256_maskstore_ps(dstPtrTempRow, k_n, dstPtrTempRow_n);

        srcPtrTemp += srcDescPtr->strides.wStride;
        dstPtrTemp += dstDescPtr->strides.wStride;
    }
}

void normalize_2D_tensor_axis3(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, Rpp32f *dstPtr, RpptDescPtr dstDescPtr,
                         Rpp32f *meanPtr, Rpp32f *invStdDevPtr, Rpp32f shift, Rpp32u *dims, Rpp32u *paramStride)
{
    Rpp32f *srcPtrTemp = srcPtr;
    Rpp32f *dstPtrTemp = dstPtr;
    Rpp32u totalLength = dims[0]*dims[1];
    Rpp32u rem = (totalLength)%PACK;
    Rpp32u v_n = (!rem) ? totalLength/PACK: (totalLength/PACK)+1;

    __m256 meanptr_n = _mm256_set1_ps(meanPtr[0]);
    __m256 invStdDevPtr_n = _mm256_set1_ps(invStdDevPtr[0]);
    __m256 shift_n = _mm256_set1_ps(shift);


    __m256 srcPtrTempRow_n;

    for(Rpp32u j = 0; j < (v_n - 1); j++) {
        srcPtrTempRow_n = _mm256_loadu_ps(srcPtrTemp);
        __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
        _mm256_storeu_ps(dstPtrTemp, dstPtrTempRow_n);
        srcPtrTemp += PACK;
        dstPtrTemp += PACK;
    }
    __m256i k_n;

    if (rem != 0) {
        k_n = maskGenerate_epi32((int32_t)2147483648, rem);
        shift_n = maskGenerate_ps(shift, rem);
        meanptr_n = maskGenerate_ps(meanPtr[0], rem);
        invStdDevPtr_n = maskGenerate_ps(invStdDevPtr[0], rem);
    } else {
        k_n = _mm256_set1_epi32((int32_t)2147483648);
    }

    srcPtrTempRow_n = _mm256_maskload_ps(srcPtrTemp, k_n);
    __m256 dstPtrTempRow_n = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(srcPtrTempRow_n, meanptr_n), invStdDevPtr_n), shift_n);
    _mm256_maskstore_ps(dstPtrTemp, k_n, dstPtrTempRow_n);
}

RppStatus normalize_audio_host_tensor(Rpp32f* srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp32f* dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      Rpp32s *srcLengthTensor,
                                      Rpp32s *channelsTensor,
                                      Rpp32s axis_mask,
                                      Rpp32f mean,
                                      Rpp32f stdDev,
                                      Rpp32f scale,
                                      Rpp32f shift,
                                      Rpp32f epsilon,
                                      Rpp32s ddof,
                                      Rpp32u numOfDims)
{
	omp_set_dynamic(0);
#pragma omp parallel for num_threads(srcDescPtr->n)
	for(int batchCount = 0; batchCount < srcDescPtr->n; batchCount++)
	{
        Rpp32f *srcPtrTemp = srcPtr + batchCount * srcDescPtr->strides.nStride;
		Rpp32f *dstPtrTemp = dstPtr + batchCount * dstDescPtr->strides.nStride;
        Rpp32u srcAudioDims[numOfDims], srcReductionDims[numOfDims], srcStride[numOfDims], paramStride[numOfDims];
        srcAudioDims[0] = srcLengthTensor[batchCount];

        srcAudioDims[1] = channelsTensor[batchCount];
        if (axis_mask == 3) {
            srcStride[0] = srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = 1;
            srcReductionDims[1] = srcAudioDims[0] * srcAudioDims[1];
            paramStride[0] = paramStride[1] = 0;
        } else if (axis_mask == 1) {
            srcStride[0] = srcDescPtr->strides.wStride;
            srcStride[1] = srcDescPtr->strides.cStride;
            srcReductionDims[0] = srcAudioDims[1];
            srcReductionDims[1] = srcAudioDims[0];
            paramStride[0] = 1;
            paramStride[1] = 0;
        } else if (axis_mask == 2) {
            srcStride[0] = srcDescPtr->strides.cStride;
            srcStride[1] = srcDescPtr->strides.wStride;
            srcReductionDims[0] = srcAudioDims[0];
            srcReductionDims[1] = srcAudioDims[1];
            paramStride[0] = 0;
            paramStride[1] = 1;
        }

        Rpp32f* meanTensor = (Rpp32f *)malloc(srcReductionDims[0] * sizeof(Rpp32f));
        Rpp32f* stdDevTensor = (Rpp32f *)malloc(srcReductionDims[0] * sizeof(Rpp32f));

        meanTensor[0] = mean;
        stdDevTensor[0] = stdDev;
        if(!mean) {
            if (axis_mask == 1)
                compute_2D_mean_axis1(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
            else if (axis_mask == 2)
                compute_2D_mean_axis2_4samples(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
            else if (axis_mask == 3)
                compute_2D_mean_axis3(srcPtrTemp, meanTensor, srcReductionDims, srcStride);
        }
        if(!stdDev) {
            if (axis_mask == 1)
                compute_2D_inv_std_dev_axis1(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
            else if (axis_mask == 2)
                compute_2D_inv_std_dev_axis2(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
            else if (axis_mask == 3)
                compute_2D_inv_std_dev_axis3(srcPtrTemp, meanTensor, stdDevTensor, srcReductionDims, srcStride);
        }
        if (axis_mask == 1)
            normalize_2D_tensor_cpu(srcPtrTemp, srcDescPtr, dstPtrTemp, dstDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);
        else if (axis_mask == 2)
            normalize_2D_tensor_cpu(srcPtrTemp, srcDescPtr, dstPtrTemp, dstDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);
        else if (axis_mask == 3)
            normalize_2D_tensor_axis3(srcPtrTemp, srcDescPtr, dstPtrTemp, dstDescPtr, meanTensor, stdDevTensor, shift, srcAudioDims, paramStride);

        // No mean and std dev
        // No mean
        // No std dev
        // Mean and std dev
        // Axis : 0
        // Axis : 1
        // if(mean && stdDev)
        // {
        //     for(int d = 0; d < numOfDims; d++)
        //     {
        //         #pragma omp simd
        //         for(int i = 0; i < srcAudioDims[d]; i++)
        //         {
        //             *dstPtrTemp = (*srcPtrTemp - mean) * stdDev + shift;
        //         }
        //     }
        //     return RPP_SUCCESS;
        // }
        // if(!mean)
        // {
        //     for(int d = 0; d < numOfDims; d++)
        //     {
        //         #pragma omp simd
        //         for(int i = 0; i < srcAudioDims[d]; i++)
        //         {
        //             *dstPtrTemp = (*srcPtrTemp - mean) * stdDev + shift;
        //         }
        //     }
        //     return RPP_SUCCESS;
        // }
        free(meanTensor);
        free(stdDevTensor);
    }
    return RPP_SUCCESS;
}