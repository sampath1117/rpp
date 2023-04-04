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
        case 3:
            strcpy(funcName, "spectrogram");
            break;
        case 5:
            strcpy(funcName, "mel_filter_bank");
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

    // Set Width as 1 for src, dst
    maxSrcWidth = 1;
    maxDstWidth = 1;

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

        srcDims[count].height = sfinfo.frames;
        srcDims[count].width = sfinfo.channels;
        dstDims[count].height = sfinfo.frames;
        dstDims[count].width = sfinfo.channels;

        maxSrcHeight = std::max(maxSrcHeight, (int)srcDims[count].height);
        maxSrcWidth = std::max(maxSrcWidth, (int)srcDims[count].width);
        maxDstHeight = std::max(maxDstHeight, (int)srcDims[count].height);
        maxDstWidth = std::max(maxDstWidth, (int)srcDims[count].width);
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

    if(test_case == 3)
        dstDescPtr->w = 1;
    else
        dstDescPtr->w = maxChannels;

    srcDescPtr->c = 1;
    dstDescPtr->c = 1;

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
    count = 0;
    dr = opendir(src);
    Rpp32f *audio_temp = (Rpp32f *)calloc(srcDescPtr->h * srcDescPtr->w, sizeof(Rpp32f));
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
            readcount = (int) sf_read_float (infile, audio_temp, bufferLength);
            if(readcount != bufferLength)
                std::cerr<<"F32 Unable to read audio file completely"<<std::endl;

            Rpp32f *audio_temp_ptr = audio_temp;
            if(srcDims[count].width == 1)
                memcpy(input_temp_f32, audio_temp_ptr, srcDims[count].height * sizeof (Rpp32f));
            else
            {
                for(int i = 0; i < srcDims[count].height; i++)
                {
                    memcpy(input_temp_f32, audio_temp_ptr, srcDims[count].width * sizeof (Rpp32f));
                    audio_temp_ptr += srcDims[count].width;
                    input_temp_f32 += srcDescPtr->strides.hStride;
                }
            }
        }

        i++;
        count++;

        // Close input
        sf_close (infile);
    }
    closedir(dr);
    free(audio_temp);

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

    RpptImagePatch *d_srcDims;
    hipMalloc(&d_srcDims, noOfAudioFiles * sizeof(RpptImagePatch));

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

            hipMemcpy(d_srcDims, srcDims, noOfAudioFiles * sizeof(RpptImagePatch), hipMemcpyHostToDevice);
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_to_decibels_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_srcDims, cutOffDB, multiplier, referenceMagnitude, handle);
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

            hipMemcpy(d_srcDims, srcDims, noOfAudioFiles * sizeof(RpptImagePatch), hipMemcpyHostToDevice);

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_pre_emphasis_filter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, d_srcDims, coeff, borderType, handle);
            }
            else
                missingFuncFlag = 1;

            hipMemcpy(outputf32, d_outputf32, ioBufferSizeInBytes_f32, hipMemcpyDeviceToHost);
            verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
            break;
        }
        case 3:
        {
            test_case_name = "spectrogram";

            bool centerWindows = true;
            bool reflectPadding = true;
            Rpp32f *windowFn = NULL;
            Rpp32s power = 2;
            Rpp32s windowLength = 512;
            Rpp32s windowStep = 256;
            Rpp32s nfft = windowLength;
            RpptSpectrogramLayout layout = RpptSpectrogramLayout::FT;

            int windowOffset = 0;
            if(!centerWindows)
                windowOffset = windowLength;

            maxDstWidth = 0;
            maxDstHeight = 0;
            if(layout == RpptSpectrogramLayout::FT)
            {
                for(int i = 0; i < noOfAudioFiles; i++)
                {
                    dstDims[i].height = nfft / 2 + 1;
                    dstDims[i].width = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                    maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                    maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                }
            }
            else
            {
                for(int i = 0; i < noOfAudioFiles; i++)
                {
                    dstDims[i].height = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                    dstDims[i].width = nfft / 2 + 1;
                    maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                    maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                }
            }

            dstDescPtr->w = maxDstWidth;
            dstDescPtr->h = maxDstHeight;

            dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
            dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
            dstDescPtr->strides.wStride = 1;
            dstDescPtr->strides.cStride = 1;

            // Set buffer sizes for src/dst
            unsigned long long spectrogramBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
            outputf32 = (Rpp32f *)realloc(outputf32, spectrogramBufferSize * sizeof(Rpp32f));
            float* d_outputf32;
            hipMalloc(&d_outputf32, spectrogramBufferSize * sizeof(Rpp32f));

            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_spectrogram_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, srcLengthTensor, centerWindows, reflectPadding, windowFn, nfft, power, windowLength, windowStep, layout, handle);
            }
            else
                missingFuncFlag = 1;

            hipMemcpy(outputf32, d_outputf32, spectrogramBufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost);
            verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
            break;
        }
        case 5:
        {
            test_case_name = "mel_filter_bank";
            Rpp32f sampleRate = 16000;
            Rpp32f minFreq = 0.0;
            Rpp32f maxFreq = sampleRate / 2;
            RpptMelScaleFormula melFormula = RpptMelScaleFormula::SLANEY;
            Rpp32s numFilter = 128;
            bool normalize = true;

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
                dstDims[i].height = numFilter;
                dstDims[i].width = srcDims[i].width;
                maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
            }

            srcDescPtr->h = maxSrcHeight;
            srcDescPtr->w = maxSrcWidth;
            dstDescPtr->h = maxDstHeight;
            dstDescPtr->w = maxDstWidth;

            // Optionally set w stride as a multiple of 8 for dst
            srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
            dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

            srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
            srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
            srcDescPtr->strides.wStride = srcDescPtr->c;
            srcDescPtr->strides.cStride = 1;

            dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
            dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
            dstDescPtr->strides.wStride = dstDescPtr->c;
            dstDescPtr->strides.cStride = 1;

            // Set buffer sizes for src/dst
            unsigned long long spectrogramBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
            unsigned long long melFilterBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
            inputf32 = (Rpp32f *)realloc(inputf32, spectrogramBufferSize * sizeof(Rpp32f));
            outputf32 = (Rpp32f *)realloc(outputf32, melFilterBufferSize * sizeof(Rpp32f));

            // Read source data
            read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 0, audioNames);

            float* d_inputf32, *d_outputf32;
            hipMalloc(&d_inputf32, spectrogramBufferSize * sizeof(Rpp32f));
            hipMalloc(&d_outputf32, melFilterBufferSize * sizeof(Rpp32f));
            hipMemcpy(d_inputf32, inputf32, spectrogramBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice);
            hipMemset(d_outputf32, 0.0f, melFilterBufferSize * sizeof(Rpp32f));
            
            start_omp = omp_get_wtime();
            start = clock();
            if (ip_bitDepth == 2)
            {
                rppt_mel_filter_bank_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, srcDims, maxFreq, minFreq, melFormula, numFilter, sampleRate, normalize, handle);
            }
            else
                missingFuncFlag = 1;
            
            hipMemcpy(outputf32, d_outputf32, melFilterBufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost);
            verify_output(outputf32, dstDescPtr, dstDims, test_case_name, audioNames);
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
    }
    hipFree(d_srcLengthTensor);
    hipFree(d_srcDims);

    return 0;
}
