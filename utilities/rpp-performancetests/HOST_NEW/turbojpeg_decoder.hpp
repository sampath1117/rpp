/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <turbojpeg.h>
#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>

// //! Default constructor
// //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
// /*!
//     \param input_buffer  User provided buffer containig the encoded image
//     \param input_size Size of the compressed data provided in the input_buffer
//     \param width pointer to the user's buffer to write the width of the compressed image to
//     \param height pointer to the user's buffer to write the height of the compressed image to
//     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to
// */
// void decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps);

// //! Decodes the actual image data
// /*!
//     \param input_buffer  User provided buffer containig the encoded image
//     \param output_buffer User provided buffer used to write the decoded image into
//     \param input_size Size of the compressed data provided in the input_buffer
//     \param max_decoded_width The maximum width user wants the decoded image to be. Image will be downscaled if bigger.
//     \param max_decoded_height The maximum height user wants the decoded image to be. Image will be downscaled if bigger.
//     \param original_image_width The actual width of the compressed image. decoded width will be equal to this if this is smaller than max_decoded_width
//     \param original_image_height The actual height of the compressed image. decoded height will be equal to this if this is smaller than max_decoded_height
// */
// void  decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
//                         size_t max_decoded_width, size_t max_decoded_height,
//                         size_t original_image_width, size_t original_image_height,
//                         size_t &actual_decoded_width, size_t &actual_decoded_height) ;

#define STR(X) std::string(X)

tjhandle m_jpegDecompressor;
const static unsigned SCALING_FACTORS_COUNT =  16;
const tjscalingfactor SCALING_FACTORS[SCALING_FACTORS_COUNT] = {
        { 2, 1 },
        { 15, 8 },
        { 7, 4 },
        { 13, 8 },
        { 3, 2 },
        { 11, 8 },
        { 5, 4 },
        { 9, 8 },
        { 1, 1 },
        { 7, 8 },
        { 3, 4 },
        { 5, 8 },
        { 1, 2 },
        { 3, 8 },
        { 1, 4 },
        { 1, 8 }
};
const static unsigned _max_scaling_factor = 8;


void initialize(){
    m_jpegDecompressor = tjInitDecompress();

#if 0
    int num_avail_scalings = 0;
    auto scaling_factors = tjGetScalingFactors	(&num_avail_scalings);
    for(int i = 0; i < num_avail_scalings; i++) {
        if(scaling_factors[i].num < scaling_factors[i].denom) {

            printf("%d / %d  - ",scaling_factors[i].num, scaling_factors[i].denom );
        }
    }
#endif
};

size_t read_data(std::string file_name, unsigned char* buf) //, size_t read_size)
{
    FILE* _current_fPtr;
    _current_fPtr = fopen(file_name.c_str(), "rb");// Open the file,
    if(!_current_fPtr) {
        std::cerr << "POINTER null";
        return 0;
    }


    fseek(_current_fPtr, 0 , SEEK_END);// Take the file read pointer to the end

    size_t _current_file_size = ftell(_current_fPtr);// Check how many bytes are there between and the current read pointer position (end of the file)

    if(_current_file_size == 0)
    {
        std::cerr<<"\n the file is empty Gonna close";
        // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, 0 , SEEK_SET);// Take the file pointer back to the start


    // std::cerr<<"\n _current_file_size ::"<<_current_file_size;
    // Requested read size bigger than the file size? just read as many bytes as the file size
    // read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), _current_file_size, _current_fPtr);
    return actual_read_size;
}
void decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps)
{
    //TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns the color components
    if(tjDecompressHeader2(m_jpegDecompressor,
                            input_buffer,
                            input_size,
                            width,
                            height,
                            color_comps) != 0)
    {
        // ignore "Could not determine Subsampling type error"
        if ( STR(tjGetErrorStr2(m_jpegDecompressor)).find("Could not determine subsampling type for JPEG image") == std::string::npos) {
            std::cerr<<"Jpeg header decode failed "<<STR(tjGetErrorStr2(m_jpegDecompressor));
            exit(0);
        }
    }
    // return Status::OK;
}

void decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                  size_t max_decoded_width, size_t max_decoded_height,
                                  size_t original_image_width, size_t original_image_height,
                                  size_t &actual_decoded_width, size_t &actual_decoded_height)
{
    int tjpf = TJPF_RGB;
    int planes = 3;

    if (original_image_width < max_decoded_width)
        actual_decoded_width = original_image_width;
    else
        actual_decoded_width = max_decoded_width;
    if (original_image_height < max_decoded_height)
        actual_decoded_height = original_image_height;
    else
        actual_decoded_height = max_decoded_height;

    if ( original_image_width > (_max_scaling_factor*max_decoded_width)  || original_image_height > (_max_scaling_factor*max_decoded_height))
    {
        unsigned int crop_width, crop_height;
        float in_ratio = static_cast<float>(original_image_width) / original_image_height;
        if(original_image_width > (_max_scaling_factor*max_decoded_width))
        {
            crop_width =  _max_scaling_factor*max_decoded_width;
            if (crop_width > original_image_width) crop_width = original_image_width;
            crop_height = crop_width * (1.0/in_ratio);
            if (crop_height > _max_scaling_factor*max_decoded_width) crop_height = _max_scaling_factor*max_decoded_width;
        }
        else if(original_image_height > (_max_scaling_factor*max_decoded_height))
        {
            crop_height = _max_scaling_factor*max_decoded_height;
            if (crop_height > original_image_height) crop_height = original_image_height;
            crop_width  = crop_height  * in_ratio;
            if (crop_width > _max_scaling_factor*max_decoded_width) crop_width = _max_scaling_factor*max_decoded_width;
        }
        if( tjDecompress2_partial_scale(m_jpegDecompressor,
                        input_buffer,
                        input_size,
                        output_buffer,
                        actual_decoded_width,
                        max_decoded_width * planes,
                        max_decoded_height,
                        tjpf,
                        TJFLAG_FASTDCT,
                        crop_width, crop_height) != 0)

        {
            std::cerr<<"Jpeg partial image decode failed "<<STR(tjGetErrorStr2(m_jpegDecompressor));
            exit(0);
        }
        // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
        uint scaledw = max_decoded_width, scaledh = max_decoded_height;
        for (auto scaling_factor : SCALING_FACTORS) {
            scaledw = TJSCALED(crop_width, scaling_factor);
            scaledh = TJSCALED(crop_height, scaling_factor);
            if (scaledw <= max_decoded_width && scaledh <= max_decoded_height) {
                break;
            }
        }
        actual_decoded_width = scaledw;
        actual_decoded_height = scaledh;
        //std::cout << "actual_decoded_width: " << actual_decoded_width << " actual_decoded_height: " << actual_decoded_height  << std::endl;
    }
    else {
        //TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
        if (tjDecompress2(m_jpegDecompressor,
                        input_buffer,
                        input_size,
                        output_buffer,
                        actual_decoded_width,
                        max_decoded_width * planes,
                        actual_decoded_height,
                        tjpf,
                        TJFLAG_FASTDCT) != 0) {
            std::cerr<<"KO::Jpeg image decode failed "<<STR(tjGetErrorStr2(m_jpegDecompressor));
            exit(0);
        }
        // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
        if ((actual_decoded_width != original_image_width) || (actual_decoded_height != original_image_height))
        {
            uint scaledw = actual_decoded_width, scaledh = actual_decoded_height;
            for (auto scaling_factor : SCALING_FACTORS) {
                scaledw = TJSCALED(original_image_width, scaling_factor);
                scaledh = TJSCALED(original_image_height, scaling_factor);
                if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                    break;
            }
            actual_decoded_width = scaledw;
            actual_decoded_height = scaledh;
        }
    }

    // return Status::OK;
}

void release() {
    tjDestroy(m_jpegDecompressor);
}

