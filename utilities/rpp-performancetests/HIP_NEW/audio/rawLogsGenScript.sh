#!/bin/bash

# <<<<<<<<<<<<<< DEFAULT SOURCE AND DESTINATION FOLDERS (NEED NOT CHANGE) >>>>>>>>>>>>>>

cwd=$(pwd)

# Input audio files - Eight audio files
DEFAULT_SRC_FOLDER="$cwd/../../../TEST_AUDIO_FILES/eight_samples_single_channel_src1/"

# # Inputs for Testing Downmixing
# # Input audio file - single audio file - multi channel
# DEFAULT_SRC_FOLDER="$cwd/../../../TEST_AUDIO_FILES/single_sample_multi_channel_src1/"

# # Inputs for Testing Non Silent Region Detection
# # Input audio files - three audio files - single channel
# DEFAULT_SRC_FOLDER="$cwd/../../../TEST_AUDIO_FILES/three_samples_single_channel_src1/"

# Output AUDIO_FILES
mkdir "$cwd/../../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
DEFAULT_DST_FOLDER="$cwd/../../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"

# <<<<<<<<<<<<<< FOR MANUAL OVERRIDE, JUST REPLACE AND POINT TO THE SOURCE AND DESTINATION FOLDERS HERE >>>>>>>>>>>>>>
SRC_FOLDER="$DEFAULT_SRC_FOLDER"
DST_FOLDER="$DEFAULT_DST_FOLDER"

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>
if [[ "$1" -lt 1 ]] | [[ "$1" -gt 6 ]]; then
    echo "The starting case# must be in the 1-3 range!"
    echo
    echo "The rawLogsGenScript.sh bash script runs the RPP audio unittest testsuite for AMDRPP functionalities in HIP/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScriptAudio.sh <S> <E>"
    echo "S     CASE_START (Starting case# (1-3))"
    echo "E     CASE_END (Ending case# (1-3))"
    exit 1
fi

if [[ "$2" -lt 1 ]] | [[ "$2" -gt 6 ]]; then
    echo "The ending case# must be in the 1-3 range!"
    echo
    echo "The rawLogsGenScript.sh bash script runs the RPP audio unittest testsuite for AMDRPP functionalities in HIP/OCL/HIP backends."
    echo
    echo "Syntax: ./testAllScriptAudio.sh <S> <E>"
    echo "S     CASE_START (Starting case# (1-3))"
    echo "E     CASE_END (Ending case# (1-3))"
    exit 1
fi

if (( "$#" < 3 )); then
    CASE_START="1"
    CASE_END="3"
else
    CASE_START="$1"
    CASE_END="$2"
    PROFILING_OPTION="$3"
fi

rm -rvf "$DST_FOLDER"/*
shopt -s extglob
mkdir build
cd build
rm -rvf ./*
cmake ..
make -j16

printf "\n\n\n\n\n"
echo "##########################################################################################"
echo "Running all Audio Inputs..."
echo "##########################################################################################"

printf "\n\nUsage: ./Tensor_HIP_audio <src folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:1>"

for ((case=$CASE_START;case<=$CASE_END;case++))
do
    printf "\n\n\n\n"
    echo "--------------------------------"
    printf "Running a New Functionality...\n"
    echo "--------------------------------"
    for ((bitDepth=2;bitDepth<3;bitDepth++))
    do
        printf "\n\n\nRunning New Bit Depth...\n-------------------------\n\n"
        printf "\n./Tensor_HIP_audio $SRC_FOLDER $bitDepth $case"
        if [[ "$PROFILING_OPTION" -eq 0 ]]
        then
            ./Tensor_hip_audio "$SRC_FOLDER" "$bitDepth" "$case" | tee -a "$DST_FOLDER/Tensor_hip_audio_raw_performance_log.txt"
        elif [[ "$PROFILING_OPTION" -eq 1 ]]
        then
            rocprof --basenames on --timestamp on --stats -o "$DST_FOLDER/output_case""$case""_bitDepth""$bitDepth"".csv" ./Tensor_hip_audio "$SRC_FOLDER" "$bitDepth" "$case" | tee -a "$DST_FOLDER/Tensor_hip_audio_raw_performance_log.txt"
        fi

        echo "------------------------------------------------------------------------------------------"
    done
done

# <<<<<<<<<<<<<< EXECUTION OF ALL FUNCTIONALITIES (NEED NOT CHANGE) >>>>>>>>>>>>>>