#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "/opt/rocm/rpp/include/rppi.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <half.hpp>
#include <fstream>
#include <algorithm>
#include <iterator>
#include "hip/hip_runtime_api.h"

using namespace cv;
using namespace std;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 7;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./BatchPD_hip_pkd3 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:81> <verbosity = 0/1>\n");
        return -1;
    }

    if (atoi(argv[6]) == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[3]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[4]);
        printf("\ncase number (0:81) = %s", argv[5]);
    }

    char *src = argv[1];
    char *src_second = argv[2];
    char *dst = "/media/sampath/sampath_rpp/PROF_IMAGES/BatchPD/";
    int ip_bitDepth = atoi(argv[3]);
    unsigned int outputFormatToggle = atoi(argv[4]);
    int test_case = atoi(argv[5]);

    int ip_channel = 3;

    char funcType[1000] = {"BatchPD_HIP_PKD3"};

    char funcName[1000];
    switch (test_case)
    {
        case 0:
            strcpy(funcName, "brightness");
            outputFormatToggle = 0;
            break;
        case 1:
            strcpy(funcName, "gamma_correction");
            outputFormatToggle = 0;
            break;
        case 2:
            strcpy(funcName, "blend");
            outputFormatToggle = 0;
            break;
        case 3:
            strcpy(funcName, "blur");
            outputFormatToggle = 0;
            break;
        case 4:
            strcpy(funcName, "contrast");
            outputFormatToggle = 0;
            break;
        case 5:
            strcpy(funcName, "pixelate");
            outputFormatToggle = 0;
            break;
        case 6:
            strcpy(funcName, "jitter");
            outputFormatToggle = 0;
            break;
        case 7:
            strcpy(funcName, "snow");
            outputFormatToggle = 0;
            break;
        case 8:
            strcpy(funcName, "noise");
            outputFormatToggle = 0;
            break;
        case 9:
            strcpy(funcName, "random_shadow");
            outputFormatToggle = 0;
            break;
        case 10:
            strcpy(funcName, "fog");
            outputFormatToggle = 0;
            break;
        case 11:
            strcpy(funcName, "rain");
            outputFormatToggle = 0;
            break;
        case 12:
            strcpy(funcName, "random_crop_letterbox");
            outputFormatToggle = 0;
            break;
        case 13:
            strcpy(funcName, "exposure");
            outputFormatToggle = 0;
            break;
        case 14:
            strcpy(funcName, "histogram_balance");
            outputFormatToggle = 0;
            break;
        case 15:
            strcpy(funcName, "thresholding");
            outputFormatToggle = 0;
            break;
        case 16:
            strcpy(funcName, "min");
            outputFormatToggle = 0;
            break;
        case 17:
            strcpy(funcName, "max");
            outputFormatToggle = 0;
            break;
        case 18:
            strcpy(funcName, "integral");
            outputFormatToggle = 0;
            break;
        case 19:
            strcpy(funcName, "histogram_equalization");
            outputFormatToggle = 0;
            break;
        case 20:
            strcpy(funcName, "flip");
            outputFormatToggle = 0;
            break;
        case 21:
            strcpy(funcName, "resize");
            break;
        default:
            strcpy(funcName, "test_case");
            break;
    }


    if (outputFormatToggle == 0)
    {
        strcat(funcType, "_toPKD3");
    }
    else if (outputFormatToggle == 1)
    {
        strcat(funcType, "_toPLN3");
    }

    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
    }
    else if (ip_bitDepth == 3)
    {
        strcat(funcName, "_u8_f16_");
    }
    else if (ip_bitDepth == 4)
    {
        strcat(funcName, "_u8_f32_");
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
    }
    else if (ip_bitDepth == 6)
    {
        strcat(funcName, "_u8_i8_");
    }

    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);

    int ip_bitDepth_1_cases[14] = {21, 22, 23, 24, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39};
    int ip_bitDepth_2_cases[14] = {21, 22, 23, 24, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39};
    int ip_bitDepth_3_cases[3]  = {21, 37, 38};
    int ip_bitDepth_4_cases[3]  = {21, 37, 38};
    int ip_bitDepth_5_cases[15] = {21, 22, 23, 24, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    int ip_bitDepth_6_cases[3]  = {21, 37, 38};

    bool functionality_existence;
    if (ip_bitDepth == 0)
        functionality_existence = 1;
    else if (ip_bitDepth == 1)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_1_cases), std::end(ip_bitDepth_1_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 2)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_2_cases), std::end(ip_bitDepth_2_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 3)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_3_cases), std::end(ip_bitDepth_3_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 4)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_4_cases), std::end(ip_bitDepth_4_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 5)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_5_cases), std::end(ip_bitDepth_5_cases), [&](int i) {return i == test_case;});
    else if (ip_bitDepth == 6)
        functionality_existence = std::any_of(std::begin(ip_bitDepth_6_cases), std::end(ip_bitDepth_6_cases), [&](int i) {return i == test_case;});

    if (functionality_existence == 0)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    int missingFuncFlag = 0;

    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfImages = 0;
    int batchSize = 256;

    Mat image;

    // Get number of images
    struct dirent *de;
    struct dirent *de_sub;
    DIR *dr = opendir(src);
    std::vector<std::string> imageNamesVec;
    char subname[1000]={};
    char temp_img[1000] ={};
    printf("source directory: %s ",subname);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        if (de->d_type == DT_DIR)
        {
           strcpy(subname, src);
           strcat(subname,"/");
           strcat(subname, de->d_name);

           DIR *drsub = opendir(subname);
           while((de_sub = readdir(drsub))!= NULL)
           {
                if (strcmp(de_sub->d_name, ".") == 0 || strcmp(de_sub->d_name, "..") == 0)
                    continue;

                strcpy(temp_img, subname);
                strcat(temp_img,"/");
                strcat(temp_img, de_sub->d_name);

                noOfImages += 1;
                // std::cout<<"noOfImages: "<<noOfImages<<std::endl;
                imageNamesVec.push_back(temp_img);
           }
           closedir(drsub);
        }
        else
        {
            strcpy(temp_img, src);
            strcat(temp_img,"/");
            strcat(temp_img, de->d_name);
            noOfImages += 1;
            imageNamesVec.push_back(temp_img);
        }
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
    RppiSize *srcSize = (RppiSize *)calloc(batchSize, sizeof(RppiSize));
    RppiSize *dstSize = (RppiSize *)calloc(batchSize, sizeof(RppiSize));

    // Set maxHeight, maxWidth and ROIs for src/dst
    const int images = batchSize;
    // for(int i = 0; i < (int)imageNamesVec.size(); i++)
    // {
    //     std::string currentImageName = imageNamesVec[i];
    //     image = imread(currentImageName.c_str(), 1);

    //     maxHeight = RPPMAX2(maxHeight, image.rows);
    //     maxWidth = RPPMAX2(maxWidth, image.cols);
    //     maxDstHeight = RPPMAX2(maxDstHeight, image.rows);
    //     maxDstWidth = RPPMAX2(maxDstWidth, image.cols);
    // }
    std::cout<<"calculated max width and max height"<<std::endl;

    // Uncomment for Resize upsampling
    maxWidth = 4288;
    maxHeight = 5005;
    maxDstWidth = 4288;
    maxDstHeight = 5005;

    ioBufferSize = (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel * (unsigned long long)batchSize;
    oBufferSize = (unsigned long long)maxDstHeight * (unsigned long long)maxDstWidth * (unsigned long long)ip_channel * (unsigned long long)batchSize;

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, 1);
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, 1);

    RppiSize maxSize, maxDstSize;
    maxSize.height = maxHeight;
    maxSize.width = maxWidth;
    maxDstSize.height = maxDstHeight;
    maxDstSize.width = maxDstWidth;

    unsigned long long imageDimMax = (unsigned long long)maxHeight * (unsigned long long)maxWidth * (unsigned long long)ip_channel;
    Rpp32u elementsInRowMax = maxWidth * ip_channel;

    // Convert inputs to test various other bit depths and copy to hip buffers
    int *d_input, *d_output;
    if (ip_bitDepth == 0)
    {
        hipMalloc(&d_input, ioBufferSize);
        hipMalloc(&d_output, oBufferSize);
    }

    std::cout<<std::endl<<"maxWidth, maxHeight: "<<maxWidth<<" "<<maxHeight;

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
    for (int perfRunCount = 0; perfRunCount < 2; perfRunCount++)
    {
        for(int t = 0; t < (int)imageNamesVec.size() / batchSize; t++)
        {
            std::cout<<"Iteration: "<<t<<std::endl;
            //Read the input images
            Rpp8u *offsetted_input;
            offsetted_input = input;

            for(int i = 0; i < batchSize ; i++)
            {
                Rpp8u *input_temp;
                input_temp = offsetted_input + (i * imageDimMax);
                int idx = t * batchSize + i;

                image = imread(imageNamesVec[idx].c_str(), 1);
                srcSize[i].width = image.cols;
                srcSize[i].height = image.rows;
                dstSize[i].width = image.cols;
                dstSize[i].height = image.rows;

                Rpp8u *ip_image = image.data;
                Rpp32u elementsInRow = srcSize[i].width * ip_channel;

                for (int k = 0; k < srcSize[i].height; k++)
                {
                    memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
                    ip_image += elementsInRow;
                    input_temp += elementsInRowMax;
                }
            }

            if (ip_bitDepth == 0)
            {
                hipMemcpy(d_input, input, ioBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice);
            }

            double gpu_time_used = 0;
            switch (test_case)
            {
                case 0:
                {
                    test_case_name = "brightness";

                    Rpp32f alpha[batchSize];
                    Rpp32f beta[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        alpha[i] = 1.75;
                        beta[i] = 50;
                    }

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppi_brightness_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, alpha, beta, batchSize, handle);
                    // else if (ip_bitDepth == 1)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 2)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 3)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 4)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 5)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 6)
                    //     missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 20:
                {
                    test_case_name = "flip";

                    Rpp32u flipAxis[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        flipAxis[i] = 1;
                    }

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppi_flip_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, flipAxis, noOfImages, handle);
                    // else if (ip_bitDepth == 1)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 2)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 3)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 4)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 5)
                    //     missingFuncFlag = 1;
                    // else if (ip_bitDepth == 6)
                    //     missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 21:
                {
                    test_case_name = "resize";

                    for (i = 0; i < batchSize; i++)
                    {
                        dstSize[i].height = 300;
                        dstSize[i].width = 300;
                        if (maxDstHeight < dstSize[i].height)
                            maxDstHeight = dstSize[i].height;
                        if (maxDstWidth < dstSize[i].width)
                            maxDstWidth = dstSize[i].width;

                    }
                    maxDstSize.height = maxDstHeight;
                    maxDstSize.width = maxDstWidth;

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppi_resize_u8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_output, dstSize, maxDstSize, outputFormatToggle, batchSize, handle);
                    // else if (ip_bitDepth == 1)
                    //     missingFuncFlag = 1; // rppi_resize_f16_pkd3_batchPD_gpu(d_inputf16, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
                    // else if (ip_bitDepth == 2)
                    //     rppi_resize_f32_pkd3_batchPD_gpu(d_inputf32, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
                    // else if (ip_bitDepth == 3)
                    //     missingFuncFlag = 1; // rppi_resize_u8_f16_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf16, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
                    // else if (ip_bitDepth == 4)
                    //     rppi_resize_u8_f32_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputf32, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
                    // else if (ip_bitDepth == 5)
                    //     rppi_resize_i8_pkd3_batchPD_gpu(d_inputi8, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
                    // else if (ip_bitDepth == 6)
                    //     rppi_resize_u8_i8_pkd3_batchPD_gpu(d_input, srcSize, maxSize, d_outputi8, dstSize, maxDstSize, outputFormatToggle, noOfImages, handle);
                    else
                        missingFuncFlag = 1;

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
    avg_time_used /= (factor * 2);
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;

    // Reconvert other bit depths to 8u for output display purposes

    string fileName = std::to_string(ip_bitDepth);
    ofstream outputFile (fileName + ".csv");

    // if (ip_bitDepth == 0)
    // {
    //     hipMemcpy(output, d_output, oBufferSize, hipMemcpyDeviceToHost);
    //     Rpp8u *outputTemp;
    //     outputTemp = output;

        // if (outputFile.is_open())
        // {
        //     for (int i = 0; i < oBufferSize; i++)
        //     {
        //         outputFile << (Rpp32u) *outputTemp << ",";
        //         outputTemp++;
        //     }
        //     outputFile.close();
        // }
        // else
        //     cout << "Unable to open file!";
    // }

    rppDestroyGPU(handle);

    // OpenCV dump

    mkdir(dst, 0700);
    strcat(dst, "/");
    count = 0;

    // elementsInRowMax = maxDstWidth * ip_channel;
    // if (outputFormatToggle == 0)
    // {
    //     Rpp8u *offsetted_output;
    //     offsetted_output = output;
    //     for (j = 0; j < batchSize; j++)
    //     {
    //         int height = dstSize[j].height;
    //         int width =  dstSize[j].width;

    //         int op_size = height * width * ip_channel;
    //         Rpp8u *temp_output = (Rpp8u *)calloc(op_size, sizeof(Rpp8u));
    //         Rpp8u *temp_output_row;
    //         temp_output_row = temp_output;
    //         Rpp32u elementsInRow = width * ip_channel;
    //         Rpp8u *output_row = offsetted_output + count;

    //         for (int k = 0; k < height; k++)
    //         {
    //             memcpy(temp_output_row, (output_row), elementsInRow * sizeof (Rpp8u));
    //             temp_output_row += elementsInRow;
    //             output_row += elementsInRowMax;
    //         }
    //         count += maxDstHeight * maxDstWidth * ip_channel;

    //         char temp[1000];
    //         char outName[1000];
    //         strcpy(temp, dst);
    //         strcpy(outName, "batchpd");
    //         std::string num = to_string(j);
    //         strcat(outName, num.c_str());
    //         strcat(outName, ".jpg");
    //         strcat(temp, outName);

    //         Mat mat_op_image;
    //         mat_op_image = Mat(height, width, CV_8UC3, temp_output);
    //         imwrite(temp, mat_op_image);

    //         free(temp_output);
    //     }
    // }

    // Free memory
    free(srcSize);
    free(dstSize);
    free(input);
    free(output);

    if (ip_bitDepth == 0)
    {
        hipFree(d_input);
        hipFree(d_output);
    }
    return 0;
}