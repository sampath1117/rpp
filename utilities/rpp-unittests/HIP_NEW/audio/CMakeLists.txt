cmake_minimum_required(VERSION 3.0)
project(rpp_unittests/HIP_NEW/audio)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../../cmake)
set(ROCM_PATH /opt/rocm CACHE PATH "default ROCm installation path")
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

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

find_package(hip REQUIRED)
find_package(AMDRPP QUIET)
find_library(libsnd_LIBS
    NAMES sndfile libsndfile
    PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBSND_ROOT_DIR} "/usr/local"
    PATH_SUFFIXES lib lib64)

if(NOT hip_FOUND)
	message("-- ${Yellow}Rpp_test requires hip, Found ${hip_INCLUDE_DIRS}/hip ${hip_LIBRARIES} ${ColourReset}")
endif()

if(NOT libsnd_LIBS)
    message(FATAL_ERROR "libsnd (sndfile) could not be found. Try to specify it's location with `-DLIBSND_ROOT_DIR`.")
endif()

if (hip_FOUND)
    set(COMPILER_FOR_HIP ${ROCM_PATH}/bin/hipcc)
    set(CMAKE_CXX_COMPILER ${COMPILER_FOR_HIP})

    include_directories(${HIP_INCLUDE_DIRS} ${HIP_INCLUDE_DIRS}/hip)
    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/rpp/include /usr/local/include)
    include_directories(${SndFile_INCLUDE_DIRS})
    link_directories(${SndFile_LIBRARIES_DIR} /usr/local/lib/)
    link_directories(${ROCM_PATH}/lib ${ROCM_PATH}/rpp/lib/ /usr/local/lib)

    add_executable(Tensor_hip_audio Tensor_hip_audio.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DGPU_SUPPORT=1 -DHIP_COMPILE=1 -std=c++11")
    target_link_libraries(Tensor_hip_audio ${HIP_LIBRARIES} -lamd_rpp -lsndfile ${libsnd_LIBS} pthread boost_filesystem boost_system hip::host)
endif()
