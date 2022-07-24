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
#include <experimental/filesystem>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6

using namespace std;
using half_float::half;

typedef half Rpp16f;

void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

void verify_output(Rpp32f *dstPtr, int *srcLength, int bs, string test_case, Rpp32u stride, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "ReferenceOutputs/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        int offset = i * stride;
        int matched_indices = 0;
        for(int j = 0; j < srcLength[i]; j++)
        {
            Rpp32f ref_val, out_val;
            ref_file>>ref_val;
            out_val = dstPtr[offset + j];
            if(abs(out_val - ref_val) < 1e-4)
                matched_indices += 1;
        }
        ref_file.close();
        if(matched_indices == srcLength[i])
            file_match++;
    }

    std::cerr<<std::endl<<"Results for Test case: "<<test_case<<std::endl;
    if(file_match == bs)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<bs<<" outputs are matching with reference outputs"<<std::endl;
}

void verify_non_silent_region_detection(int *detectionIndex, int *detectionLength, string test_case, int bs, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "ReferenceOutputs/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);

        Rpp32s ref_index, ref_length;
        Rpp32s out_index, out_length;
        ref_file>>ref_index;
        ref_file>>ref_length;
        out_index = detectionIndex[i];
        out_length = detectionLength[i];

        if((out_index == ref_index) && (out_length == ref_length))
            file_match += 1;
        ref_file.close();
    }
    std::cerr<<std::endl<<"Results for Test case: "<<test_case<<std::endl;
    if(file_match == bs)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<bs<<" outputs are matching with reference outputs"<<std::endl;
}

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
    std::cerr<<"src: "<<argv[0]<<std::endl;
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
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(noOfAudioFiles, sizeof(Rpp32s));
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

    // Optionally set w stride as a multiple of 8 for src
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;

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
            Rpp32s detectionIndex[noOfAudioFiles];
            Rpp32s detectionLength[noOfAudioFiles];
            Rpp32f cutOffDB[noOfAudioFiles];
            Rpp32s windowLength[noOfAudioFiles];
            Rpp32f referencePower[noOfAudioFiles];
            Rpp32s resetInterval[noOfAudioFiles];
            bool referenceMax[noOfAudioFiles];

            for (i = 0; i < noOfAudioFiles; i++)
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

            verify_non_silent_region_detection(detectionIndex, detectionLength, test_case_name, noOfAudioFiles, audioNames);
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

            verify_output(outputf32, srcLengthTensor, noOfAudioFiles, test_case_name, srcDescPtr->strides.nStride, audioNames);
            break;
        }
        case 2:
        {
            test_case_name = "pre_emphasis_filter";
            Rpp32f coeff[noOfAudioFiles];
            for (i = 0; i < noOfAudioFiles; i++)
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

            verify_output(outputf32, srcLengthTensor, noOfAudioFiles, test_case_name, srcDescPtr->strides.nStride, audioNames);
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
                rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, srcLengthTensor, channelsTensor, normalizeWeights);
            }
            else
                missingFuncFlag = 1;

            verify_output(outputf32, srcLengthTensor, noOfAudioFiles, test_case_name, srcDescPtr->strides.nStride, audioNames);
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
