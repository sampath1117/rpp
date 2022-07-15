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
#include <fstream>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6

using namespace std;
using half_float::half;

typedef half Rpp16f;

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 4;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_audio <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:84> <verbosity = 0/1>\n");
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
        case 0:
            strcpy(funcName, "non_silent_region_detection");
            break;
        case 1:
            strcpy(funcName, "to_decibels");
            break;
        case 2:
            strcpy(funcName, "pre_emphasis_filter");
            break;
        case 3:
            strcpy(funcName, "down_mixing");
            break;
        default:
            strcpy(funcName, "test_case");
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

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, srcDescPtr->n);
    clock_t start, end;
    double start_omp, end_omp;
    double cpu_time_used, omp_time_used;

    string test_case_name;
    switch (test_case)
    {
        case 0:
        {
            test_case_name = "non_silent_region_detection";
            Rpp32s detectionIndex[srcDescPtr->n];
            Rpp32s detectionLength[srcDescPtr->n];

            Rpp32f cutOffDB[srcDescPtr->n];
            Rpp32s windowLength[srcDescPtr->n];
            Rpp32f referencePower[srcDescPtr->n];
            Rpp32s resetInterval[srcDescPtr->n];
            bool referenceMax[srcDescPtr->n];

            for (i = 0; i < srcDescPtr->n; i++)
            {
                cutOffDB[i] = -60.0;
                windowLength[i] = 3;
                referencePower[i] = 1.0;
                resetInterval[i] = -1;
                referenceMax[i] = true;
            }

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_non_silent_region_detection_host(inputf32, srcDescPtr, inputAudioSize, detectionIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, referenceMax, handle);
            }
            else
                missingFuncFlag = 1;

            // Print the detection index and length
            for(int i = 0; i < srcDescPtr->n; i++)
            {
                cout<<endl<<"Audiofile: "<<audioNames[i];
                cout<<endl<<"Index, Length: "<<detectionIndex[i]<<" "<<detectionLength[i];
            }

            break;
        }
        case 1:
        {
            test_case_name = "to_decibels";
            Rpp32f cutOffDB = -200.0;
            Rpp32f multiplier = 10.0;
            Rpp32f referenceMagnitude = 0.0;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_to_decibels_host(inputf32, srcDescPtr, outputf32, srcLengthTensor, cutOffDB, multiplier, referenceMagnitude);
            }
            else
                missingFuncFlag = 1;

            // cout<<endl<<"Output in DB: "<<endl;
            // int cnt = 0;
            // for(int i = 0; i < srcLengthTensor[0]; i++)
            // {
            //     cout<<"output["<<i<<"]: "<<outputf32[i]<<endl;
            // }

            break;
        }
        case 2:
        {
            test_case_name = "pre_emphasis_filter";
            Rpp32f coeff[srcDescPtr->n];
            for (i = 0; i < srcDescPtr->n; i++)
                coeff[i] = 0.97;
            RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_pre_emphasis_filter_host(inputf32, srcDescPtr, outputf32, inputAudioSize, coeff, borderType);
            }
            else
                missingFuncFlag = 1;

            // cout<<endl<<"Output from preemphasis filter: ";
            // for(int i = 0; i < srcDescPtr->n; i++)
            // {
            //     cout<<endl<<"Audiofile: "<<audioNames[i]<<endl;
            //     for(int j = 0; j < inputAudioSize[i]; j++)
            //     {
            //         cout<<outputf32[j]<<endl;
            //     }
            //     cout<<endl;
            // }
            break;
        }
        case 3:
        {
            test_case_name = "down_mixing";
            bool normalizeWeights = false;

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                // rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, srcLengthTensor, channelsTensor, normalizeWeights);
            }
            else
                missingFuncFlag = 1;

            // Print the mono output
            // cout<<endl<<"Printing downmixed output: "<<endl;
            // for(int i = 0; i < srcDescPtr->n; i++)
            // {
            //     int idxstart = i * srcDescPtr->strides.nStride;
            //     int idxend = idxstart + srcLengthTensor[i];
            //     for(int j = idxstart; j < idxend; j++)
            //     {
            //         cout<<"out["<<j<<"]: "<<outputf32[j]<<endl;
            //     }
            //     cout<<endl;
            // }

            break;
        }
        default:
        {
            missingFuncFlag = 1;
            break;
        }
    }

    end = clock();
    end_omp = omp_get_wtime();

    if (missingFuncFlag == 1)
    {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
        return -1;
    }

    // Display measured times
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    omp_time_used = end_omp - start_omp;
    cout << "\nCPU Time - BatchPD : " << cpu_time_used;
    cout << "\nOMP Time - BatchPD : " << omp_time_used;
    printf("\n");

    rppDestroyHost(handle);

    // Free memory
    free(inputAudioSize);
    free(srcLengthTensor);
    free(channelsTensor);
    free(inputf32);
    free(outputf32);

    return 0;
}
