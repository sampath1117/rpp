#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include "/opt/rocm/rpp/include/rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half.hpp>
#include <hip/hip_fp16.h>
#include <fstream>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6
using namespace std;

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 4;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_hip_audio <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:84> <verbosity = 0/1>\n");
        return -1;
    }

    char *src = argv[1];
    int ip_bitDepth = atoi(argv[2]);
    int test_case = atoi(argv[3]);
    int ip_channel = 1;

    // Set case names
    char funcName[1000];
    switch (test_case)
    {
        case 1:
            strcpy(funcName, "to_decibels");
            break;
        case 2:
            strcpy(funcName, "pre_emphasis_filter");
            break;
    }

    // Initialize tensor descriptors
    RpptDesc srcDesc;
    RpptDescPtr srcDescPtr;
    srcDescPtr = &srcDesc;

    // Set src/dst data types in tensor descriptors
    if (ip_bitDepth == 2)
        srcDescPtr->dataType = RpptDataType::F32;

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxLength = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    static int noOfAudioFiles = 0;

    // String ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");

    char func[1000];
    strcpy(func, funcName);
    printf("\nRunning %s...", func);

    // Get number of audio files
    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfAudioFiles += 1;
    }
    closedir(dr);

    // Initialize the AudioPatch for source
    Rpp32s *inputAudioSize = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
    Rpp32u *srcLengthTensor = (Rpp32u *) calloc(noOfAudioFiles, sizeof(Rpp32u));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));

    // Set maxLength
    char audioNames[noOfAudioFiles][1000];

    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        {
            sf_close (infile);
            continue;
        }

        inputAudioSize[count] = sfinfo.frames * sfinfo.channels;
        srcLengthTensor[count] = sfinfo.frames;
        channelsTensor[count] = sfinfo.channels;
        maxLength = std::max(maxLength, inputAudioSize[count]);

        // Close input
        sf_close (infile);
        count++;
    }
    closedir(dr);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    srcDescPtr->offsetInBytes = 0;
    srcDescPtr->n = noOfAudioFiles;
    srcDescPtr->h = 1;
    srcDescPtr->w = maxLength;
    srcDescPtr->c = ip_channel;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
    unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(ioBufferSize, sizeof(Rpp32f));

    i = 0;
    dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        Rpp32f *input_temp_f32;
        input_temp_f32 = inputf32 + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(audioNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, audioNames[count]);

        SNDFILE	*infile;
        SF_INFO sfinfo;
        int	readcount;

        // The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        {
            sf_close (infile);
            continue;
        }

        int bufferLength = sfinfo.frames * sfinfo.channels;
        if(ip_bitDepth == 2)
        {
            readcount = (int) sf_read_float (infile, input_temp_f32, bufferLength);
            if(readcount != bufferLength)
                std::cerr<<"F32 Unable to read audio file completely"<<std::endl;
        }
        i++;

        // Close input
        sf_close (infile);
    }
    closedir(dr);

    int *d_inputf32, *d_outputf32;
    if (ip_bitDepth == 2)
    {
        hipMalloc(&d_inputf32, ioBufferSizeInBytes_f32);
        hipMalloc(&d_outputf32, ioBufferSizeInBytes_f32);
        hipMemcpy(d_inputf32, inputf32, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
    }

    Rpp32u *d_srcLengthTensor;
    hipMalloc(&d_srcLengthTensor, noOfAudioFiles * sizeof(Rpp32u));

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    rppCreateWithStreamAndBatchSize(&handle, stream, noOfAudioFiles);

    clock_t start, end;
    double start_omp, end_omp;
    double gpu_time_used, omp_time_used;

    string test_case_name;
    switch (test_case)
    {
        case 1:
        {
            test_case_name = "to_decibels";
            Rpp32f cutOffDB = -200.0;
            Rpp32f multiplier = 10.0;
            Rpp32f referenceMagnitude = 0.0;

            hipMemcpy(d_srcLengthTensor, srcLengthTensor, noOfAudioFiles * sizeof(Rpp32u), hipMemcpyHostToDevice);

            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_to_decibels_gpu(d_inputf32, srcDescPtr, d_outputf32, d_srcLengthTensor, cutOffDB, multiplier, referenceMagnitude, handle);
            }
            else
                missingFuncFlag = 1;


            hipMemcpy(outputf32, d_outputf32, ioBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
            cout<<endl<<"Output in DB: "<<endl;
            int cnt = 0;
            for(int i = 0; i < srcLengthTensor[0]; i++)
            {
                cout<<"output["<<i<<"]: "<<outputf32[i]<<endl;
            }

            cout<<endl;

            break;
        }
        case 2:
        {
            test_case_name = "pre_emphasis_filter";
            Rpp32f coeff[noOfAudioFiles];
            for (i = 0; i < noOfAudioFiles; i++)
                coeff[i] = 0.97;
            RpptAudioBorderType borderType = RpptAudioBorderType::ZERO;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_pre_emphasis_filter_gpu(d_inputf32, srcDescPtr, d_outputf32, inputAudioSize, coeff, borderType, handle);
            }
            else
                missingFuncFlag = 1;

            hipMemcpy(outputf32, d_outputf32, ioBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
            cout<<endl<<"Output in DB: "<<endl;
            int cnt = 0;
            for(int i = 0; i < srcLengthTensor[0]; i++)
            {
                cout<<"output["<<i<<"]: "<<outputf32[i]<<endl;
            }

            cout<<endl;
            break;
        }
        default:
        {
            missingFuncFlag = 1;
            break;
        }
    }

    hipDeviceSynchronize();
    end = clock();

    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    // Display measured times
    gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    cout << "\nGPU Time - Tensor: " << gpu_time_used;
    printf("\n");

    rppDestroyGPU(handle);

    // Free memory
    free(inputAudioSize);
    free(srcLengthTensor);
    free(channelsTensor);
    free(inputf32);
    free(outputf32);
    if(ip_bitDepth == 2)
    {
        hipFree(d_inputf32);
        hipFree(d_outputf32);
        hipFree(d_srcLengthTensor);
    }

    return 0;
}
