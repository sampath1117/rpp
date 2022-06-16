#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "/opt/rocm/rpp/include/rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <hip/hip_fp16.h>
#include <fstream>

using namespace cv;
using namespace std;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

std::string get_interpolation_type(unsigned int val, RpptInterpolationType &interpolationType)
{
    switch(val)
    {
        case 0:
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            return "NearestNeighbor";
        }
        case 2:
        {
            interpolationType = RpptInterpolationType::BICUBIC;
            return "Bicubic";
        }
        case 3:
        {
            interpolationType = RpptInterpolationType::LANCZOS;
            return "Lanczos";
        }
        case 4:
        {
            interpolationType = RpptInterpolationType::TRIANGULAR;
            return "Triangular";
        }
        case 5:
        {
            interpolationType = RpptInterpolationType::GAUSSIAN;
            return "Gaussian";
        }
        default:
        {
            interpolationType = RpptInterpolationType::BILINEAR;
            return "Bilinear";
        }
    }
}

std::string get_noise_type(unsigned int val)
{
    switch(val)
    {
        case 0: return "SaltAndPepper";
        case 1: return "Gaussian";
        case 2: return "Shot";
        default:return "SaltAndPepper";
    }
}

int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 7;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_hip_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:84> <verbosity = 0/1>\n");
        return -1;
    }

    char *src = argv[1];
    char *src_second = argv[2];
    char *dst = "/media/sampath/sampath_rpp/PROF_IMAGES/Tensor/";

    int ip_bitDepth = atoi(argv[3]);
    unsigned int outputFormatToggle = atoi(argv[4]);
    int test_case = atoi(argv[5]);

    bool additionalParamCase = (test_case == 8 || test_case == 24 || test_case == 40 || test_case == 41 || test_case == 49);
    bool kernelSizeCase = (test_case == 40 || test_case == 41 || test_case == 49);
    bool interpolationTypeCase = (test_case == 24);
    bool noiseTypeCase = (test_case == 8);

    unsigned int verbosity = additionalParamCase ? atoi(argv[7]) : atoi(argv[6]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[6]) : 1;

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[3]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[4]);
        printf("\ncase number (0:84) = %s", argv[5]);
    }

    int ip_channel = 3;

    // Set case names

    char funcType[1000] = {"Tensor_HIP_PKD3"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        break;
    case 1:
        strcpy(funcName, "gamma_correction");
        break;
    case 2:
        strcpy(funcName, "blend");
        break;
    case 4:
        strcpy(funcName, "contrast");
        break;
    case 8:
        strcpy(funcName, "noise");
        break;
    case 13:
        strcpy(funcName, "exposure");
        break;
    case 20:
        strcpy(funcName, "flip");
        break;
    case 21:
        strcpy(funcName, "resize");
        break;
    case 24:
        strcpy(funcName, "warp_affine");
        break;
    case 31:
        strcpy(funcName, "color_cast");
        break;
    case 36:
        strcpy(funcName, "color_twist");
        break;
    case 37:
        strcpy(funcName, "crop");
        break;
    case 38:
        strcpy(funcName, "crop_mirror_normalize");
        break;
    case 40:
        strcpy(funcName, "erode");
        break;
    case 41:
        strcpy(funcName, "dilate");
        break;
    case 49:
        strcpy(funcName, "box_filter");
        break;
    case 83:
        strcpy(funcName, "gridmask");
        break;
    case 84:
        strcpy(funcName, "spatter");
        break;
    default:
        strcpy(funcName, "test_case");
        break;
    }

    // Initialize tensor descriptors

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst layouts in tensor descriptors

    if (outputFormatToggle == 0)
    {
        strcat(funcType, "_toPKD3");
        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NHWC;
    }
    else if (outputFormatToggle == 1)
    {
        strcat(funcType, "_toPLN3");
        srcDescPtr->layout = RpptLayout::NHWC;
        dstDescPtr->layout = RpptLayout::NCHW;
    }

    // Set src/dst data types in tensor descriptors

    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        strcat(funcName, "_u8_f16_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        strcat(funcName, "_u8_f32_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        strcat(funcName, "_u8_i8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }

    // Other initializations

    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfImages = 0;
    int batchSize = 3;
    Mat image;

    // String ops on function name

    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if (kernelSizeCase)
    {
        char additionalParam_char[2];
        std::sprintf(additionalParam_char, "%u", additionalParam);
        strcat(func, "_kSize");
        strcat(func, additionalParam_char);
    }
    else if (interpolationTypeCase)
    {
        std::string interpolationTypeName;
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        strcat(func, "_interpolationType");
        strcat(func, interpolationTypeName.c_str());
    }
    else if (noiseTypeCase)
    {
        std::string noiseTypeName;
        noiseTypeName = get_noise_type(additionalParam);
        strcat(func, "_noiseType");
        strcat(func, noiseTypeName.c_str());
    }

    // Get number of images
    struct dirent *de;
    DIR *dr = opendir(src);
    std::vector<std::string> imageNamesVec;
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
        imageNamesVec.push_back(de->d_name);
    }
    closedir(dr);
    
    std::string last_img_name = imageNamesVec[noOfImages - 1];
    int remImages = 0;

    //If total number of images is not a multiple of batchSize
    if((noOfImages % batchSize)!= 0)
    {
        remImages = batchSize - noOfImages % batchSize;
    }
    
    //Replicate last image for remImages
    for(int i = 0; i < remImages; i++)
    {
        imageNamesVec.push_back(last_img_name);
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc = (RpptROI *) calloc(batchSize, sizeof(RpptROI));
    RpptROI *roiTensorPtrDst = (RpptROI *) calloc(batchSize, sizeof(RpptROI));

    RpptROI *d_roiTensorPtrSrc, *d_roiTensorPtrDst;
    hipMalloc(&d_roiTensorPtrSrc, batchSize * sizeof(RpptROI));
    hipMalloc(&d_roiTensorPtrDst, batchSize * sizeof(RpptROI));

    // Initialize the ImagePatch for source and destination
    RpptImagePatch *srcImgSizes = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));
    RpptImagePatch *dstImgSizes = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));

    RpptImagePatch *d_srcImgSizes, *d_dstImgSizes;
    hipMalloc(&d_srcImgSizes, batchSize * sizeof(RpptImagePatch));
    hipMalloc(&d_dstImgSizes, batchSize * sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst
    const int images = batchSize;
    for(int i = 0; i < (int)imageNamesVec.size(); i++)
    {
        std::string currentImageName = imageNamesVec[i];
        char tempName[1000] ={};
        strcat(tempName, src);
        strcat(tempName, "/");
        strcat(tempName, currentImageName.c_str()); 
        image = imread(tempName, 1);

        maxHeight = RPPMAX2(maxHeight, image.rows);
        maxWidth = RPPMAX2(maxWidth, image.cols);
        maxDstHeight = RPPMAX2(maxDstHeight, image.rows);
        maxDstWidth = RPPMAX2(maxDstWidth, image.cols);
    }

    // Uncomment for Resize upsampling
    maxDstWidth = 400;
    maxDstHeight = 520;

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = batchSize;
    srcDescPtr->h = maxHeight;
    srcDescPtr->w = maxWidth;
    srcDescPtr->c = ip_channel;

    dstDescPtr->n = batchSize;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;
    dstDescPtr->c = ip_channel;

    std::cout<<std::endl<<"maxWidth, maxHeight: "<<maxWidth<<" "<<maxHeight;

    // Optionally set w stride as a multiple of 8 for src/dst
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = ip_channel * srcDescPtr->w;
    srcDescPtr->strides.wStride = ip_channel;
    srcDescPtr->strides.cStride = 1;

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = ip_channel * dstDescPtr->w;
        dstDescPtr->strides.wStride = ip_channel;
        dstDescPtr->strides.cStride = 1;
    }
    else if (dstDescPtr->layout == RpptLayout::NCHW)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;
    }

    // Set buffer sizes in pixels for src/dst
    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)batchSize;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)batchSize;

    // Set buffer sizes in bytes for src/dst (including offsets)
    unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u *input = (Rpp8u *)calloc(ioBufferSizeInBytes_u8, 1);
    Rpp8u *output = (Rpp8u *)calloc(oBufferSizeInBytes_u8, 1);

    // Convert inputs to test various other bit depths and copy to hip buffers
    int *d_input, *d_output;
    if (ip_bitDepth == 0)
    {
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_output, oBufferSizeInBytes_u8);
    }
 
    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, batchSize);

    clock_t start, end;
    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;

    string test_case_name;
    printf("\nRunning %s 100 times (each time with a batch size of %d images) and computing mean statistics...\n", func, batchSize);

    std::cout<<"iterations: "<<(int)imageNamesVec.size() / batchSize<<std::endl;
    for (int perfRunCount = 0; perfRunCount < 100; perfRunCount++)
    {
        for(int t = 0; t < (int)imageNamesVec.size() / batchSize; t++)
        {
            //Read the input images
            Rpp8u *offsetted_input;
            offsetted_input = input + srcDescPtr->offsetInBytes;

            for(int i = 0; i < batchSize ; i++)
            {
                Rpp8u *input_temp;
                input_temp = offsetted_input + (i * srcDescPtr->strides.nStride);
                int idx = t * batchSize + i;
                
                char temp[1000];
                strcpy(temp, src1);
                strcat(temp, imageNamesVec[idx].c_str());
                
                image = imread(temp, 1); 
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = image.cols;
                roiTensorPtrSrc[i].xywhROI.roiHeight = image.rows;

                roiTensorPtrDst[i].xywhROI.xy.x = 0;
                roiTensorPtrDst[i].xywhROI.xy.y = 0;
                roiTensorPtrDst[i].xywhROI.roiWidth = image.cols;
                roiTensorPtrDst[i].xywhROI.roiHeight = image.rows;

                srcImgSizes[i].width = roiTensorPtrSrc[count].xywhROI.roiWidth;
                srcImgSizes[i].height = roiTensorPtrSrc[count].xywhROI.roiHeight;
                dstImgSizes[i].width = roiTensorPtrDst[count].xywhROI.roiWidth;
                dstImgSizes[i].height = roiTensorPtrDst[count].xywhROI.roiHeight;

                Rpp8u *ip_image = image.data;
                Rpp32u elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth * ip_channel;

                for (int k = 0; k < roiTensorPtrSrc[i].xywhROI.roiHeight; k++)
                {
                    memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
                    ip_image += elementsInRow;
                    input_temp += srcDescPtr->strides.hStride;
                }
            }

            if (ip_bitDepth == 0)
            {   
                //Memcpy to HIP buffers
                hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
                hipMemcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, batchSize * sizeof(RpptROI), hipMemcpyHostToDevice);
            }
                    
            double gpu_time_used;
            switch (test_case)
            {
                case 0:
                {
                    test_case_name = "brightness";
                    Rpp32f alpha[batchSize];
                    Rpp32f beta[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        alpha[i] = 1.75;
                        beta[i] = 50;
                    }

                    start = clock();

                    if (ip_bitDepth == 0)
                    {   
                        rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    }
                    // else if (ip_bitDepth == 1)
                    //     rppt_brightness_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 2)
                    //     rppt_brightness_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 3)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 4)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 5)
                    //     rppt_brightness_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, alpha, beta, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 6)
                    //     missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;
                
                    break;
                }
                case 20:
                {
                    test_case_name = "flip";

                    Rpp32u horizontalFlag[images];
                    Rpp32u verticalFlag[images];
                    for (i = 0; i < images; i++)
                    {
                        horizontalFlag[i] = 1;
                        verticalFlag[i] = 0;
                    }

                    // Uncomment to run test case with an xywhROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                        roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                        roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                    }*/

                    // Uncomment to run test case with an ltrbROI override
                    /*for (i = 0; i < images; i++)
                        roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                        roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                        roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                        roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                    }
                    roiTypeSrc = RpptRoiType::LTRB;
                    roiTypeDst = RpptRoiType::LTRB;*/

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppt_flip_gpu(d_input, srcDescPtr, d_output, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 1)
                    //     rppt_flip_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 2)
                    //     rppt_flip_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 3)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 4)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 5)
                    //     rppt_flip_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, horizontalFlag, verticalFlag, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 6)
                    //     missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 21:
                {
                    test_case_name = "resize";

                    if (interpolationType != RpptInterpolationType::BILINEAR)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0; i < images; i++)
                    {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = 400;//roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = 520;//roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                    }

                    // Uncomment to run test case with an xywhROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                    }*/

                    // Uncomment to run test case with an ltrbROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                        roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                        roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                        roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                    }
                    roiTypeSrc = RpptRoiType::LTRB;
                    roiTypeDst = RpptRoiType::LTRB;*/

                    hipMemcpy(d_dstImgSizes, dstImgSizes, batchSize * sizeof(RpptImagePatch), hipMemcpyHostToDevice);

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 1)
                    //     rppt_resize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 2)
                    //     rppt_resize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 3)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 4)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 5)
                    //     rppt_resize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, d_dstImgSizes, interpolationType, d_roiTensorPtrSrc, roiTypeSrc, handle);
                    // else if (ip_bitDepth == 6)
                    //     missingFuncFlag = 1;
                    // else
                    //     missingFuncFlag = 1;

                    break;
                }
                default:
                    missingFuncFlag = 1;
                    break;
            }

            // std::cout<<"Came out of the Switch case"<<std::endl;

            hipDeviceSynchronize();
            end = clock();

            // std::cout<<"Done hip device syncronize!"<<std::endl;

            if (missingFuncFlag == 1)
            {
                printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
                return -1;
            }

            // Display measured times

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            if (gpu_time_used > max_time_used)
                max_time_used = gpu_time_used;
            if (gpu_time_used < min_time_used)
                min_time_used = gpu_time_used;
            avg_time_used += gpu_time_used;
        }
    }

    int factor = (int)imageNamesVec.size() / batchSize;
    avg_time_used /= (factor * 100);
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;
    
    // Reconvert other bit depths to 8u for output display purposes

    string fileName = std::to_string(ip_bitDepth);
    ofstream outputFile (fileName + ".csv");

    if (ip_bitDepth == 0)
    {
        hipMemcpy(output, d_output, oBufferSizeInBytes_u8, hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = output + dstDescPtr->offsetInBytes;

        if (outputFile.is_open())
        {
            for (int i = 0; i < oBufferSize; i++)
            {
                outputFile << (Rpp32u) *outputTemp << ",";
                outputTemp++;
            }
            outputFile.close();
        }
        else
            cout << "Unable to open file!";
    }

    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault;
    roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI.xy.x = 0;
    roiPtrDefault->xywhROI.xy.y = 0;
    roiPtrDefault->xywhROI.roiWidth = dstDescPtr->w;
    roiPtrDefault->xywhROI.roiHeight = dstDescPtr->h;

    for (int i = 0; i < batchSize; i++)
    {
        roiTensorPtrSrc[i].xywhROI.roiWidth = RPPMIN2(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrSrc[i].xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.roiWidth);
        roiTensorPtrSrc[i].xywhROI.roiHeight = RPPMIN2(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrSrc[i].xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.roiHeight);
        roiTensorPtrSrc[i].xywhROI.xy.x = RPPMAX2(roiPtrDefault->xywhROI.xy.x, roiTensorPtrSrc[i].xywhROI.xy.x);
        roiTensorPtrSrc[i].xywhROI.xy.y = RPPMAX2(roiPtrDefault->xywhROI.xy.y, roiTensorPtrSrc[i].xywhROI.xy.y);
    }

    rppDestroyGPU(handle);

    // OpenCV dump

    mkdir(dst, 0700);
    strcat(dst, "/");
    count = 0;

    if(dstDescPtr->layout == RpptLayout::NHWC)
    {
        Rpp8u *offsetted_output;
        offsetted_output = output + dstDescPtr->offsetInBytes;
        for (j = 0; j < batchSize; j++)
        {
            int height = dstImgSizes[j].height;
            int width = dstImgSizes[j].width;

            int op_size = height * width * ip_channel;
            Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
            Rpp8u *temp_output_row;
            temp_output_row = temp_output;
            Rpp32u elementsInRow = width * ip_channel;
            Rpp8u *output_row = offsetted_output + count;

            for (int k = 0; k < height; k++)
            {
                memcpy(temp_output_row, (output_row), elementsInRow * sizeof (Rpp8u));
                temp_output_row += elementsInRow;
                output_row += dstDescPtr->strides.hStride;
            }
            count += dstDescPtr->strides.nStride;

            char temp[1000];
            strcpy(temp, dst);
            strcat(temp, imageNamesVec[j].c_str());
            // printf("image name is: %s ",temp);

            Mat mat_op_image;
            mat_op_image = Mat(height, width, CV_8UC3, temp_output);
            imwrite(temp, mat_op_image);

            free(temp_output);
        }
    }

    // Free memory

    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    hipFree(d_roiTensorPtrSrc);
    hipFree(d_roiTensorPtrDst);
    hipFree(d_srcImgSizes);
    hipFree(d_dstImgSizes);
    free(input);
    free(output);
    free(srcImgSizes);
    free(dstImgSizes);

    if (ip_bitDepth == 0)
    {
        hipFree(d_input);
        hipFree(d_output);
    }
    return 0;
}