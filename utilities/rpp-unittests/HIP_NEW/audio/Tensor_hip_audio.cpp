#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <iostream>
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <half/half.hpp>

#include <hip/hip_fp16.h>
#include <fstream>
#include <experimental/filesystem>
#include <iomanip>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6
using namespace std;

void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

void verify_output(Rpp32f *dstPtr, RpptDescPtr dstDescPtr, RpptImagePatchPtr dstDims, string test_case, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HIP_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int batchcount = 0; batchcount < dstDescPtr->n; batchcount++)
    {
        string current_file_name = audioNames[batchcount];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }
        int matched_indices = 0;
        Rpp32f ref_val, out_val;
        Rpp32f *dstPtrCurrent = dstPtr + batchcount * dstDescPtr->strides.nStride;
        Rpp32f *dstPtrRow = dstPtrCurrent;
        for(int i = 0; i < dstDims[batchcount].height; i++)
        {
            Rpp32f *dstPtrTemp = dstPtrRow;
            for(int j = 0; j < dstDims[batchcount].width; j++)
            {
                ref_file>>ref_val;
                out_val = dstPtrTemp[j];
                bool invalid_comparision = ((out_val == 0.0f) && (ref_val != 0.0f));
                std::cerr<<ref_val<<", "<<out_val<<std::endl;
                if(!invalid_comparision && abs(out_val - ref_val) < 1e-2)
                    matched_indices += 1;
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
        ref_file.close();
        if(matched_indices == (dstDims[batchcount].width * dstDims[batchcount].height) && matched_indices !=0)
            file_match++;
    }

    std::cerr<<std::endl<<"Results for Test case: "<<test_case<<std::endl;
    if(file_match == dstDescPtr->n)
        std::cerr<<"PASSED!"<<std::endl;
    else
        std::cerr<<"FAILED! "<<file_match<<"/"<<dstDescPtr->n<<" outputs are matching with reference outputs"<<std::endl;
}

void verify_non_silent_region_detection(int *detectedIndex, int *detectionLength, string test_case, int bs, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HIP_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";
    int file_match = 0;
    for (int i = 0; i < bs; i++)
    {
        string current_file_name = audioNames[i];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + "_ref_" + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }

        Rpp32s ref_index, ref_length;
        Rpp32s out_index, out_length;
        ref_file>>ref_index;
        ref_file>>ref_length;
        out_index = detectedIndex[i];
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

void read_from_text_files(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, RpptImagePatch *srcDims, string test_case, int read_type, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    string pattern = "HIP_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "REFERENCE_OUTPUTS_AUDIO/";

    string read_type_str;
    if(read_type == 0)
        read_type_str = "_ref_";
    else
        read_type_str = "_info_";

    for (int batchcount = 0; batchcount < srcDescPtr->n; batchcount++)
    {
        string current_file_name = audioNames[batchcount];
        size_t last_index = current_file_name.find_last_of(".");
        current_file_name = current_file_name.substr(0, last_index);  // Remove extension from file name
        string out_file = ref_path + test_case + "/" + test_case + read_type_str + current_file_name + ".txt";
        ref_file.open(out_file, ios::in);
        if(!ref_file.is_open())
        {
            cerr<<"Unable to open the file specified! Please check the path of the file given as input"<<endl;
            break;
        }

        if(read_type == 0)
        {
            Rpp32f ref_val;
            Rpp32f *srcPtrCurrent = srcPtr + batchcount * srcDescPtr->strides.nStride;
            Rpp32f *srcPtrRow = srcPtrCurrent;
            for(int i = 0; i < srcDims[batchcount].height; i++)
            {
                Rpp32f *srcPtrTemp = srcPtrRow;
                for(int j = 0; j < srcDims[batchcount].width; j++)
                {
                    ref_file>>ref_val;
                    srcPtrTemp[j] = ref_val;
                }
                srcPtrRow += srcDescPtr->strides.hStride;
            }
        }
        else
        {
            Rpp32s ref_height, ref_width;
            ref_file>>ref_height;
            ref_file>>ref_width;
            srcDims[batchcount].height = ref_height;
            srcDims[batchcount].width = ref_width;
        }
        ref_file.close();
    }
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 3;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_hip_audio <src folder> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:3>\n");
        return -1;
    }

    char *src = argv[1];
    int ip_bitDepth = atoi(argv[2]);
    int test_case = atoi(argv[3]);

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
        case 8:
            strcpy(funcName, "normalize");
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

    // Set src/dst data types in tensor descriptors
    if (ip_bitDepth == 2)
    {
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxChannels = 0;
    int maxLength = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    unsigned long long count = 0;
    unsigned long long iBufferSize = 0;
    unsigned long long oBufferSize = 0;
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
    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(noOfAudioFiles, sizeof(RpptImagePatch));

    // Set maxLength
    char audioNames[noOfAudioFiles][1000];

    // Set Height as 1 for src, dst
    maxSrcHeight = 1;
    maxDstHeight = 1;

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
        maxLength = std::max(maxLength, srcLengthTensor[count]);
        srcDims[count].width = sfinfo.frames;
        dstDims[count].width = sfinfo.frames;
        srcDims[count].height = 1;
        dstDims[count].height = 1;

        maxSrcWidth = std::max(maxSrcWidth, srcLengthTensor[count]);
        maxDstWidth = std::max(maxDstWidth, srcLengthTensor[count]);
        maxChannels = std::max(maxChannels, channelsTensor[count]);

        // Close input
        sf_close (infile);
        count++;
    }
    closedir(dr);

    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfAudioFiles;
    dstDescPtr->n = noOfAudioFiles;

    srcDescPtr->h = maxSrcHeight;
    dstDescPtr->h = maxDstHeight;

    srcDescPtr->w = maxSrcWidth;
    dstDescPtr->w = maxDstWidth;

    srcDescPtr->c = maxChannels;
    if(test_case == 3)
        dstDescPtr->c = 1;
    else
        dstDescPtr->c = maxChannels;

    // Optionally set w stride as a multiple of 8 for src
    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst
    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
    srcDescPtr->strides.wStride = srcDescPtr->c;
    srcDescPtr->strides.cStride = 1;

    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
    dstDescPtr->strides.wStride = dstDescPtr->c;
    dstDescPtr->strides.cStride = 1;

    // Set buffer sizes for src/dst
    iBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;

    // Initialize host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));
    unsigned long long ioBufferSizeInBytes_f32 = (iBufferSize * 4) + srcDescPtr->offsetInBytes;

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

    Rpp32s *d_srcLengthTensor;
    hipMalloc(&d_srcLengthTensor, noOfAudioFiles * sizeof(Rpp32s));

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

            hipMemcpy(d_srcLengthTensor, srcLengthTensor, noOfAudioFiles * sizeof(Rpp32s), hipMemcpyHostToDevice);

            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_to_decibels_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_srcLengthTensor, cutOffDB, multiplier, referenceMagnitude, handle);
            }
            else
                missingFuncFlag = 1;


            hipMemcpy(outputf32, d_outputf32, ioBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
            verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
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
                rppt_pre_emphasis_filter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, inputAudioSize, coeff, borderType, handle);
            }
            else
                missingFuncFlag = 1;

            hipMemcpy(outputf32, d_outputf32, ioBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
            verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
            break;
        }
        case 8:
        {
            test_case_name = "normalize";
            Rpp32s axis_mask = 1;
            Rpp32f mean, std_dev, scale, shift, epsilon;
            mean = std_dev = scale = shift = epsilon = 0.0f;
            Rpp32s ddof = 0;
            Rpp32s num_of_dims = 2;

            Rpp32s srcDimsTensor[noOfAudioFiles * 2];

            // Read source dimension
            read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 1, audioNames);

            maxDstHeight = 0;
            maxDstWidth = 0;
            maxSrcHeight = 0;
            maxSrcWidth = 0;
            for(int i = 0; i < noOfAudioFiles; i++)
            {
                maxSrcHeight = std::max(maxSrcHeight, (int)srcDims[i].height);
                maxSrcWidth = std::max(maxSrcWidth, (int)srcDims[i].width);
            }
            maxDstHeight = maxSrcHeight;
            maxDstWidth = maxSrcWidth;

            for (i = 0; i < noOfAudioFiles * 2; i += 2)
            {
                srcDimsTensor[i] = (int)srcDims[i / 2].height;
                srcDimsTensor[i + 1] = (int)srcDims[i / 2].width;
            }

            srcDescPtr->h = maxSrcHeight;
            srcDescPtr->w = maxSrcWidth;
            dstDescPtr->h = maxDstHeight;
            dstDescPtr->w = maxDstWidth;

            srcDescPtr->c = 1;
            dstDescPtr->c = 1;

            srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
            srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
            srcDescPtr->strides.wStride = maxSrcWidth;
            srcDescPtr->strides.cStride = 1;

            dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
            dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
            dstDescPtr->strides.wStride = maxSrcWidth;
            dstDescPtr->strides.cStride = 1;

            // // Set buffer sizes for src/dst
            unsigned long long spectrogramBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
            unsigned long long padBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
            inputf32 = (Rpp32f *)realloc(inputf32, spectrogramBufferSize * sizeof(Rpp32f));
            outputf32 = (Rpp32f *)realloc(outputf32, padBufferSize * sizeof(Rpp32f));

            int *d_inputf32_test, *d_outputf32_test;
            hipMalloc(&d_inputf32_test, spectrogramBufferSize * sizeof(Rpp32f));
            hipMalloc(&d_outputf32_test, padBufferSize * sizeof(Rpp32f));

            // Read source data
            read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 0, audioNames);

            hipMemcpy(d_inputf32_test, inputf32, spectrogramBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_normalize_audio_gpu(d_inputf32_test, srcDescPtr, d_outputf32_test, dstDescPtr, srcDimsTensor, axis_mask,
                                          mean, std_dev, scale, shift, epsilon, ddof, num_of_dims, handle);
            }
            else
                missingFuncFlag = 1;



            hipMemcpy(outputf32, d_outputf32_test, ioBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
            verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
            hipFree(d_inputf32_test);
            hipFree(d_outputf32_test);

            // verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
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
