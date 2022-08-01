cmake_minimum_required(VERSION 3.0)
project(rpp_performancetests/HOST_NEW/audio)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../../cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(BoldBlue    "${Esc}[1;34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
endif()

find_package(AMDRPP QUIET)
find_library(libsnd_LIBS
    NAMES sndfile libsndfile
    PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBSND_ROOT_DIR} "/usr/local"
    PATH_SUFFIXES lib lib64)

if(NOT libsnd_LIBS)
    message(FATAL_ERROR "libsnd (sndfile) could not be found. Try to specify it's location with `-DLIBSND_ROOT_DIR`.")
else()
    message("-- ${Green}${PROJECT_NAME} set to build with rpp and libsnd (sndfile) ${ColourReset}")
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/rpp/include /usr/local/include)
    include_directories(${SndFile_INCLUDE_DIRS})
    link_directories(${SndFile_LIBRARIES_DIR} /usr/local/lib/)
    link_directories(${ROCM_PATH}/lib ${ROCM_PATH}/rpp/lib/ /usr/local/lib)

    add_executable(Tensor_host_audio Tensor_host_audio.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DOCL_COMPILE=1 -DRPP_BACKEND_OPENCL=1 -std=c++11")
    target_link_libraries(Tensor_host_audio -lamd_rpp ${libsnd_LIBS} -lsndfile pthread boost_filesystem boost_system)
endif()
