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
#include <fstream>
#include <vector>
#include <math.h>
#include <random>

// Include this header file to use functions from libsndfile
#include <sndfile.h>

// libsndfile can handle more than 6 channels but we'll restrict it to 6
#define	MAX_CHANNELS 6

using namespace std;
using half_float::half;

typedef half Rpp16f;

inline double Hann(double x) {
    return 0.5 * (1 + std::cos(x * M_PI));
}

inline void windowed_sinc(ResamplingWindow &window,
        int coeffs, int lobes, std::function<double(double)> envelope = Hann) {
    Rpp32f scale = 2.0f * lobes / (coeffs - 1);
    Rpp32f scale_envelope = 2.0f / coeffs;
    window.coeffs = coeffs;
    window.lobes = lobes;
    window.lookup.clear();
    window.lookup.resize(coeffs + 5);
    window.lookup_size = window.lookup.size();
    window.pxLookupMax = _mm_set1_epi32(window.lookup_size - 2);
    int center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
        Rpp32f x = (i - center) * scale;
        Rpp32f y = (i - center) * scale_envelope;
        Rpp32f w = sinc(x) * envelope(y);
        window.lookup[i + 1] = w;
    }
    window.center = center + 1;
    window.scale = 1 / scale;
}

void listFiles(const std::string &path, std::vector<std::string>& audioNameVec) {
    if (auto dir = opendir(path.c_str())) {
        while (auto f = readdir(dir)) {
            if (!f->d_name || f->d_name[0] == '.') continue;
            if (f->d_type == DT_DIR)
                listFiles(path + f->d_name + "/", audioNameVec);

            if (f->d_type == DT_REG)
                audioNameVec.push_back(path + f->d_name);
        }
        closedir(dir);
    }
}

void remove_substring(string &str, string &pattern)
{
    std::string::size_type i = str.find(pattern);
    while (i != std::string::npos)
    {
        str.erase(i, pattern.length());
        i = str.find(pattern, i);
   }
}

void read_from_text_files(Rpp32f *srcPtr, RpptDescPtr srcDescPtr, RpptImagePatch *srcDims, string test_case, int read_type, char audioNames[][1000])
{
    fstream ref_file;
    string ref_path = get_current_dir_name();
    // string pattern = "HOST_NEW/audio/build";
    string pattern = "rpp-performancetests/HOST_NEW/audio/build";
    remove_substring(ref_path, pattern);
    ref_path = ref_path + "rpp-unittest/REFERENCE_OUTPUTS_AUDIO/";

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
        printf("\nUsage: ./Tensor_host_audio <src folder> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:3>\n");
        return -1;
    }

    char *src = argv[1];
    int ip_bitDepth = atoi(argv[2]);
    int test_case = atoi(argv[3]);

    Rpp32f quality = 50.0f;
    ResamplingWindow window;
    int lobes = std::round(0.007 * quality * quality - 0.09 * quality + 3);
    int lookupSize = lobes * 64 + 1;
    windowed_sinc(window, lookupSize, lobes);

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
        case 4:
            strcpy(funcName, "slice");
            break;
        case 5:
            strcpy(funcName, "mel_filter_bank");
            break;
        case 6:
            strcpy(funcName, "spectrogram");
            break;
        case 7:
            strcpy(funcName, "resample");
            break;
        case 8:
            strcpy(funcName, "normalize");
            break;
        case 9:
            strcpy(funcName, "pad");
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
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    unsigned long long count = 0;
    unsigned long long iBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfAudioFiles = 0;
    int batchSize = 192;

    char func[1000];
    strcpy(func, funcName);

    // Get number of audio files
    struct dirent *de;
    struct dirent *de_sub;
    DIR *dr = opendir(src);
    DIR *dr1 = opendir(src);

    std::vector<std::string> AudioNameVec;


    char subname[1000]={};
    char temp_img[1000] ={};
    listFiles(src, AudioNameVec);

    // while ((de = readdir(dr)) != NULL)
    // {
    //     if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
    //         continue;
    //     if (de->d_type == DT_DIR)
    //     {
    //        strcpy(subname, src);
    //        strcat(subname,"/");
    //        strcat(subname, de->d_name);

    //        DIR *drsub = opendir(subname);
    //        while((de_sub = readdir(drsub))!= NULL)
    //        {
    //             if (strcmp(de_sub->d_name, ".") == 0 || strcmp(de_sub->d_name, "..") == 0)
    //                 continue;

    //             strcpy(temp_img, subname);
    //             strcat(temp_img,"/");
    //             strcat(temp_img, de_sub->d_name);

    //             noOfAudioFiles += 1;
    //             AudioNameVec.push_back(temp_img);
    //        }
    //        closedir(drsub);
    //     }
    //     else
    //     {
    //         strcpy(temp_img, src);
    //         strcat(temp_img,"/");
    //         strcat(temp_img, de->d_name);
    //         noOfAudioFiles += 1;
    //         AudioNameVec.push_back(temp_img);

    //     }
    // }
    // closedir(dr);
    // std::cerr<<"AudioNameVec  "<<AudioNameVec.size();
    // for(int i=0 ; i<AudioNameVec.size();i++)
    // {
    //     std::cerr<<AudioNameVec[i]<<"\n";
    // }
    noOfAudioFiles = AudioNameVec.size();
    std::string last_img_name = AudioNameVec[noOfAudioFiles - 1];
    int remImages = 0;

    //If total number of images is not a multiple of batchSize
    if((noOfAudioFiles % batchSize)!= 0)
    {
        remImages = batchSize - noOfAudioFiles % batchSize;
    }

    //Replicate last image for remImages
    for(int i = 0; i < remImages; i++)
    {
        AudioNameVec.push_back(last_img_name);
    }
    noOfAudioFiles = AudioNameVec.size();


    // Initialize the AudioPatch for source
    Rpp32s *inputAudioSize = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));

    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));

    if(test_case==5 || test_case==9)
    {
        RpptImagePatch *dstDims1 = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));
    }

    // Set maxLength
    char audioNames[batchSize][1000];

    // Set Height as 1 for src, dst
    maxSrcWidth = 1;
    maxDstWidth = 1;


    dr = opendir(src);
    std::cerr << "SIZE :: " << AudioNameVec.size() << "\n";
    for (int j=0; j< AudioNameVec.size();j++)
    {
        char temp[1000];
        strcpy(temp, AudioNameVec[j].c_str());
        SNDFILE	*infile= NULL;
        SF_INFO sfinfo;
        int	readcount;

        //The SF_INFO struct must be initialized before using it
        memset (&sfinfo, 0, sizeof (sfinfo));
        // if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
        // {
        //     sf_close (infile);
        //     continue;
        // }
        srcLengthTensor[0] = 522320;//sfinfo.frames;
        channelsTensor[0] = 1;//sfinfo.channels;

        // maxSrcHeight = std::max(maxSrcWidth, srcLengthTensor[0]);
        // maxDstHeight = std::max(maxDstWidth, srcLengthTensor[0]);
        // maxChannels = std::max(maxChannels, channelsTensor[0]);
        maxSrcHeight = std::max(maxSrcHeight, (int)srcLengthTensor[0]);
        maxDstHeight = std::max(maxDstHeight, (int)srcLengthTensor[0]);
        maxSrcWidth = std::max(maxSrcWidth, (int)channelsTensor[0]);
        maxDstWidth = std::max(maxDstWidth, (int)channelsTensor[0]);
        // Close input
        sf_close (infile);

        // count++;
    }
    closedir(dr);


    // Set numDims, offset, n/c/h/w values for src/dst
    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 0;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = batchSize;
    dstDescPtr->n = batchSize;

    srcDescPtr->h = maxSrcHeight;
    dstDescPtr->h = maxDstHeight;

    srcDescPtr->w = maxSrcWidth;
    dstDescPtr->w = maxDstWidth;

    srcDescPtr->c = 1;
    dstDescPtr->c = 1;
    // if(test_case == 3)
    //     dstDescPtr->c = 1;
    // else
    //     dstDescPtr->c = 1;

    // Optionally set w stride as a multiple of 8 for src
    // srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    // dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

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
    Rpp32f *inputf32_second = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    std::random_device rd;     // Only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
    std::uniform_real_distribution<double> uni(0.85,1.15); // Guaranteed unbiased


    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    rppCreateWithBatchSize(&handle, batchSize);
    printf("\nRunning %s 100 times (each time with a batch size of %d audio files) and computing mean statistics...", func, noOfAudioFiles);

    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;
    double cascaded_time_used = 0;
    string test_case_name;
    int numRuns = 1;

    for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
    {
        for(int t = 0; t < (int)AudioNameVec.size() / batchSize; t++)
        {
            std::cerr<<"iteration: "<<t<<std::endl;
            clock_t start, end;
            double start_omp, end_omp;
            double cpu_time_used, omp_time_used;
            double decode_omp_time_used = 0, resample_omp_time_used = 0, nsr_omp_time_used = 0, slice_omp_time_used = 0, preemph_omp_time_used = 0;
            double decode_start_omp = 0, decode_end_omp = 0;
            double resample_start_omp = 0, resample_end_omp = 0;
            double nsr_start_omp = 0, nsr_end_omp = 0;
            double slice_start_omp = 0, slice_end_omp = 0;
            double preemph_start_omp = 0, preemph_end_omp = 0;

            start_omp = omp_get_wtime();
            decode_start_omp = omp_get_wtime();
            for(int i = 0; i < batchSize ; i++)
            {
                int idx = t * batchSize + i;
                Rpp32f *input_temp_f32;
                input_temp_f32 = inputf32 + (i * srcDescPtr->strides.nStride);

                SNDFILE	*infile;
                SF_INFO sfinfo;
                int	readcount;
                char temp[1000];
                strcpy(temp, AudioNameVec[idx].c_str());

                // The SF_INFO struct must be initialized before using it
                memset (&sfinfo, 0, sizeof (sfinfo));
                if (!(infile = sf_open (temp, SFM_READ, &sfinfo)) || sfinfo.channels > MAX_CHANNELS)
                {
                    sf_close (infile);
                    continue;
                }
                inputAudioSize[i] = sfinfo.frames * sfinfo.channels;
                srcLengthTensor[i] = sfinfo.frames;
                channelsTensor[i] = sfinfo.channels;

                // std::cerr<<"srcLengthTensor[i]: "<<sfinfo.frames<<std::endl;
                int bufferLength = sfinfo.frames * sfinfo.channels;

                if(ip_bitDepth == 2)
                {
                    readcount = (int) sf_read_float (infile, input_temp_f32, bufferLength);
                    if(readcount != bufferLength)
                        std::cerr<<"F32 Unable to read audio file completely"<<std::endl;
                }

                sf_close(infile);
            }

            // rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, false);
            // end = clock();
            end_omp = omp_get_wtime();
            omp_time_used = end_omp - start_omp;

            decode_end_omp = omp_get_wtime();
            decode_omp_time_used = decode_end_omp - decode_start_omp;
            switch (test_case)
            {
                case 0:
                {
                    test_case_name = "non_silent_region_detection";
                    Rpp32f detectedIndex[batchSize];
                    Rpp32f detectionLength[batchSize];
                    Rpp32f cutOffDB = 60.0;
                    Rpp32s windowLength = 2048;
                    Rpp32f referencePower = 0.0f;
                    Rpp32s resetInterval = 8192;

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_non_silent_region_detection_host(inputf32, srcDescPtr, inputAudioSize, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval);
                    }
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 1:
                {
                    test_case_name = "to_decibels";
                    Rpp32f cutOffDB = log(1e-20);
                    Rpp32f multiplier = log(10);
                    Rpp32f referenceMagnitude = 1.0;

                    for (int e = 0; e < batchSize; e++)
                    {
                        srcDims[e].height = srcLengthTensor[0];
                        srcDims[e].width = 1;
                    }

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_to_decibels_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDims, cutOffDB, multiplier, referenceMagnitude);
                    }
                    else
                        missingFuncFlag = 1;
                    break;
                }
                case 2:
                {
                    test_case_name = "pre_emphasis_filter";
                    Rpp32f *coeff = NULL;
                    RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_pre_emphasis_filter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, inputAudioSize, coeff, borderType);
                    }
                    else
                        missingFuncFlag = 1;

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
                        rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, normalizeWeights);
                    }
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 4:
                {
                    test_case_name = "slice";

                    Rpp32f* fillValues = NULL;
                    Rpp32s numDims = 1;
                    Rpp32s srcDimsTensor[batchSize * numDims];
                    Rpp32f anchor[batchSize * numDims];
                    Rpp32f shape[batchSize * numDims];
                    Rpp32f cutOffDB = 60.0;
                    Rpp32s windowLength = 2048;
                    Rpp32f referencePower = 0.0f;
                    Rpp32s resetInterval = 8192;

                // for (i = 0, j = i * 2; i < noOfAudioFiles; i++, j += 2)
                // {
                //     srcDimsTensor[j] = srcLengthTensor[i];
                //     srcDimsTensor[j + 1] = 1;
                //     shape[j] =  dstDims[i].width = 5;
                //     shape[j + 1] = dstDims[i].height = 1;
                //     anchor[j] = 2;
                //     anchor[j + 1] = 0;
                // }
                // fillValues[0] = 0.5f;

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_non_silent_region_detection_host(inputf32, srcDescPtr, inputAudioSize, anchor, shape, cutOffDB, windowLength, referencePower, resetInterval);
                        for (i = 0; i < batchSize; i++) {
                            srcDimsTensor[i] = srcLengthTensor[i];
                            dstDims[i].width = shape[i];
                            dstDims[i].height = 1;
                        }
                        rppt_slice_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, anchor, shape, fillValues);
                    }
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 5:
                {
                    test_case_name = "mel_filter_bank";

                    bool centerWindows = true;
                    bool reflectPadding = true;
                    Rpp32f *windowFn = NULL;
                    Rpp32s power = 2;
                    Rpp32s windowLength = 320;
                    Rpp32s windowStep = 160;
                    Rpp32s nfft = 512;
                    RpptSpectrogramLayout layout = RpptSpectrogramLayout::FT;

                    Rpp32f sampleRate = 16000;
                    Rpp32f minFreq = 0.0;
                    Rpp32f maxFreq = 0.0;
                    RpptMelScaleFormula melFormula = RpptMelScaleFormula::SLANEY;
                    Rpp32s numFilter = 80;
                    bool normalize = true;
                    int windowOffset = 0;
                    if(!centerWindows)
                        windowOffset = windowLength;
                    // Read source dimension
                    // read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 1, audioNames);
                    maxDstHeight = 0;
                    maxDstWidth = 0;
                    maxSrcHeight = 0;
                    maxSrcWidth = 0;
                    for(int i = 0; i < batchSize; i++)
                        {
                            dstDims[i].height = nfft / 2 + 1;
                            dstDims[i].width = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                            maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                            maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                        }
                    dstDescPtr->w = maxDstWidth;
                    dstDescPtr->h = maxDstHeight;

                    // Optionally set w stride as a multiple of 8 for dst
                    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

                    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
                    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
                    dstDescPtr->strides.wStride = dstDescPtr->c;
                    dstDescPtr->strides.cStride = 1;

                    // Set buffer sizes for src/dst
                    unsigned long long spectrogramBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
                    inputf32_second = (Rpp32f *)realloc(inputf32_second, spectrogramBufferSize * sizeof(Rpp32f));
                    start_omp = omp_get_wtime();
                    start = clock();
                    rppt_spectrogram_host(inputf32, srcDescPtr, inputf32_second, dstDescPtr, srcLengthTensor, centerWindows, reflectPadding, windowFn, nfft, power, windowLength, windowStep, layout);


                    ///*********** akielsh melfilter********************
                    maxSrcHeight = maxDstHeight;
                    maxSrcWidth = maxDstWidth;

                    maxSrcHeight = maxDstHeight;
                    maxSrcWidth = maxDstWidth;
                    maxDstHeight = numFilter;

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
                    unsigned long long melFilterBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
                    // inputf32 = (Rpp32f *)realloc(inputf32, melFilterBufferSize * sizeof(Rpp32f));
                    outputf32 = (Rpp32f *)realloc(outputf32, melFilterBufferSize * sizeof(Rpp32f));


                    // Read source data
                    // read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 0, audioNames);

                    if (ip_bitDepth == 2)
                    {
                        rppt_mel_filter_bank_host(inputf32_second, srcDescPtr, outputf32, dstDescPtr, dstDims, maxFreq, minFreq, melFormula, numFilter, sampleRate, normalize);
                    }
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 6:
                {
                    test_case_name = "spectrogram";

                    bool centerWindows = true;
                    bool reflectPadding = true;
                    Rpp32f *windowFn = NULL;
                    Rpp32s power = 2;
                    Rpp32s windowLength = 320;
                    Rpp32s windowStep = 160;
                    Rpp32s nfft = 512;
                    RpptSpectrogramLayout layout = RpptSpectrogramLayout::FT;

                    int windowOffset = 0;
                    if(!centerWindows)
                        windowOffset = windowLength;

                    maxDstWidth = 0;
                    maxDstHeight = 0;
                    if(layout == RpptSpectrogramLayout::FT)
                    {
                        for(int i = 0; i < batchSize; i++)
                        {
                            dstDims[i].height = nfft / 2 + 1;
                            dstDims[i].width = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                            maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                            maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                        }
                    }
                    else
                    {
                        for(int i = 0; i < batchSize; i++)
                        {
                            dstDims[i].height = ((srcLengthTensor[i] - windowOffset) / windowStep) + 1;
                            dstDims[i].width = nfft / 2 + 1;
                            maxDstHeight = std::max(maxDstHeight, (int)dstDims[i].height);
                            maxDstWidth = std::max(maxDstWidth, (int)dstDims[i].width);
                        }
                    }

                    dstDescPtr->w = maxDstWidth;
                    dstDescPtr->h = maxDstHeight;

                    // Optionally set w stride as a multiple of 8 for dst
                    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

                    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
                    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
                    dstDescPtr->strides.wStride = dstDescPtr->c;
                    dstDescPtr->strides.cStride = 1;

                    // Set buffer sizes for src/dst
                    unsigned long long spectrogramBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
                    outputf32 = (Rpp32f *)realloc(outputf32, spectrogramBufferSize * sizeof(Rpp32f));

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_spectrogram_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, centerWindows, reflectPadding, windowFn, nfft, power, windowLength, windowStep, layout);
                    }
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 7:
                {
                    test_case_name = "resample";

                    Rpp32f inRateTensor[batchSize];
                    Rpp32f outRateTensor[batchSize];
                    Rpp32s resampleLengthTensor[batchSize];

                    maxDstWidth = 0;
                    for(int cc = 0; cc < batchSize; cc++)
                    {
                        inRateTensor[cc] = 16000;
                        auto random_val = uni(rng);
                        outRateTensor[cc] = 18400;//random_val * inRateTensor[cc];
                        channelsTensor[cc] = 1;
                        Rpp32f scaleRatio = outRateTensor[cc] / inRateTensor[cc];
                        resampleLengthTensor[cc] = std::ceil(scaleRatio * srcLengthTensor[cc]);
                        maxDstWidth = std::max(maxDstWidth, resampleLengthTensor[cc]);
                    }

                    dstDescPtr->h = 1;
                    dstDescPtr->w = maxDstWidth;
                    dstDescPtr->c = 1;

                    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
                    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
                    dstDescPtr->strides.wStride = 1;
                    dstDescPtr->strides.cStride = 1;

                    // Set buffer sizes for dst
                    unsigned long long resampleBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;

                    // // Initialize host buffers for output
                    outputf32 = (Rpp32f *)realloc(outputf32, sizeof(Rpp32f) * resampleBufferSize);

                    resample_start_omp = omp_get_wtime();
                    if (ip_bitDepth == 2)
                        rppt_resample_host(inputf32, srcDescPtr, outputf32, dstDescPtr, inRateTensor, outRateTensor, srcLengthTensor, channelsTensor, quality, window);
                    else
                        missingFuncFlag = 1;
                    resample_end_omp = omp_get_wtime();

//                    // call NSR
//                    Rpp32f detectedIndex[batchSize] = {0.0f};
//                    Rpp32f detectionLength[batchSize] = {0.0f};
//                    Rpp32f cutOffDB = -60.0;
//                    Rpp32s windowLength = 2048;
//                    Rpp32f referencePower = 0.0f;
//                    Rpp32s resetInterval = 8192;

                    nsr_start_omp = omp_get_wtime();
//                    if (ip_bitDepth == 2)
//                        rppt_non_silent_region_detection_host(outputf32, dstDescPtr, resampleLengthTensor, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval);
//                    else
//                        missingFuncFlag = 1;
                    nsr_end_omp = omp_get_wtime();


                    // call slice
//                    bool slice_enable = true;
//                    Rpp32s srcDimsTensor[batchSize * 2];
//                    Rpp32f anchor[batchSize];
//                    Rpp32f shape[batchSize];
//                    Rpp32f fillValues[batchSize];
//
//                    for (int cc = 0, j = 0; cc < batchSize; cc++, j += 2) {
//                        srcDimsTensor[j] = resampleLengthTensor[cc];
//                        srcDimsTensor[j + 1] = 1;
//                        anchor[cc] = detectedIndex[cc];
//                        shape[cc] = detectionLength[cc];
//                        fillValues[cc] = 0.0f;
//                    }
//
//                    unsigned long long sliceBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
//                    Rpp32f *slicef32 = (Rpp32f *)malloc(sliceBufferSize * sizeof(Rpp32f));

                    slice_start_omp = omp_get_wtime();
//                    if (ip_bitDepth == 2)
//                    {
//                        rppt_slice_host(outputf32, dstDescPtr, slicef32, dstDescPtr, resampleLengthTensor, anchor, shape, fillValues);
//                    }
//                    else
//                        missingFuncFlag = 1;
                    slice_end_omp = omp_get_wtime();


                    // call pre emphasis
//                    Rpp32f coeff[batchSize];
//                    Rpp32s preemphLengthTensor[batchSize];
//                    RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;
//                    for(int cc = 0; cc < batchSize; cc++)
//                    {
//                        coeff[cc] = 0.97;
//                        preemphLengthTensor[cc] = (int)shape[cc];
//                    }

//                    Rpp32f *preemphf32 = (Rpp32f *)malloc(sliceBufferSize * sizeof(Rpp32f));
                    preemph_start_omp = omp_get_wtime();
//                    if (ip_bitDepth == 2)
//                    {
//                        rppt_pre_emphasis_filter_host(slicef32, dstDescPtr, preemphf32, dstDescPtr, preemphLengthTensor, coeff, borderType);
//                    }
//                    else
//                        missingFuncFlag = 1;
                   preemph_end_omp = omp_get_wtime();

//                    free(slicef32);
//                    free(preemphf32);

                    break;
                }
                case 8:
                {
                    test_case_name = "normalize";
                    Rpp32s axis_mask = 2;
                    Rpp32f mean, std_dev, scale, shift, epsilon;
                    mean = std_dev =  shift = epsilon = 0.0f;
                    scale = 1.0f;
                    Rpp32s ddof = 0;
                    Rpp32s num_of_dims = 2;

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_normalize_audio_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, channelsTensor, axis_mask,
                                                mean, std_dev, scale, shift, epsilon, ddof, num_of_dims);
                    }
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 9:
                {
                    test_case_name = "pad";
                    Rpp32f anchor[noOfAudioFiles * 2];
                    Rpp32f shape[noOfAudioFiles * 2];
                    Rpp32f fillValues[noOfAudioFiles];
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
                        shape[i] = maxSrcHeight;
                        shape[i + 1] = maxSrcWidth;
                        anchor[i] = 0.0f;
                        anchor[i + 1] = 0.0f;
                        fillValues[i / 2] = 40.0f;
                        dstDims[i].height = maxSrcHeight;
                        dstDims[i].width = maxSrcWidth;
                    }

                    srcDescPtr->h = maxSrcHeight;
                    srcDescPtr->w = 1;
                    dstDescPtr->h = maxDstHeight;
                    dstDescPtr->w = 1;

                    srcDescPtr->c = maxSrcWidth;
                    dstDescPtr->c = maxDstWidth;

                    srcDescPtr->strides.nStride = srcDescPtr->c * srcDescPtr->w * srcDescPtr->h;
                    srcDescPtr->strides.hStride = srcDescPtr->c * srcDescPtr->w;
                    srcDescPtr->strides.wStride = maxSrcWidth;
                    srcDescPtr->strides.cStride = 1;

                    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;
                    dstDescPtr->strides.hStride = dstDescPtr->c * dstDescPtr->w;
                    dstDescPtr->strides.wStride = maxDstWidth;
                    dstDescPtr->strides.cStride = 1;

                    // // Set buffer sizes for src/dst
                    unsigned long long spectrogramBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)srcDescPtr->c * (unsigned long long)srcDescPtr->n;
                    unsigned long long padBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)dstDescPtr->c * (unsigned long long)dstDescPtr->n;
                    inputf32 = (Rpp32f *)realloc(inputf32, spectrogramBufferSize * sizeof(Rpp32f));
                    outputf32 = (Rpp32f *)realloc(outputf32, padBufferSize * sizeof(Rpp32f));

                    // Read source data
                    read_from_text_files(inputf32, srcDescPtr, srcDims, "spectrogram", 0, audioNames);

                    start_omp = omp_get_wtime();
                    start = clock();
                    if (ip_bitDepth == 2)
                    {
                        rppt_slice_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDimsTensor, anchor, shape, fillValues);
                    }
                    else
                        missingFuncFlag = 1;

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

            cpu_time_used += ((double)(end - start)) / CLOCKS_PER_SEC;
            omp_time_used += end_omp - start_omp;

            resample_omp_time_used = resample_end_omp - resample_start_omp;
            nsr_omp_time_used = nsr_end_omp - nsr_start_omp;
            slice_omp_time_used = slice_end_omp - slice_start_omp;
            preemph_omp_time_used = preemph_end_omp - preemph_start_omp;
            // std::cerr<<"resample time: "<<resample_omp_time_used<<std::endl;
            // std::cerr<<"nsr time: "<<nsr_omp_time_used<<std::endl;
            // std::cerr<<"slice time: "<<slice_omp_time_used<<std::endl;
            // std::cerr<<"preemph time: "<<preemph_omp_time_used<<std::endl;

            double total_time = resample_omp_time_used + nsr_omp_time_used + slice_omp_time_used + preemph_omp_time_used;
            cascaded_time_used += resample_omp_time_used;

            if (total_time > max_time_used)
                max_time_used = total_time;
            if (total_time < min_time_used)
                min_time_used = total_time;
            avg_time_used += resample_omp_time_used;
        }
    }

    int factor = 1;
    avg_time_used /= numRuns;
    cascaded_time_used /= numRuns;
    // avg_time_used /= 100;

    // Display measured times
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << cascaded_time_used << endl;
    rppDestroyHost(handle);

    // Free memory
    free(inputAudioSize);
    free(srcLengthTensor);
    free(channelsTensor);
    free(srcDims);
    free(dstDims);
    free(inputf32);
    free(outputf32);

    return 0;
}
