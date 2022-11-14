import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case_start', type=str, default='0', help='Testing range starting case # - (1-3)')
parser.add_argument('--case_end', type=str, default='9', help='Testing range ending case # - (1-3)')
parser.add_argument('--profiling', type=str, default='NO', help='Run with profiler? - (YES/NO)')
args = parser.parse_args()

profilingOption = args.profiling
caseStart = args.case_start
caseEnd = args.case_end

if caseEnd < caseStart:
    print("Ending case# must be greater than starting case#. Aborting!")
    exit(0)

if caseStart < "1" or caseStart > "3":
    print("Starting case# must be in the 1-3 range. Aborting!")
    exit(0)

if caseEnd < "1" or caseEnd > "3":
    print("Ending case# must be in the 1-3 range. Aborting!")
    exit(0)

if profilingOption == "NO":
    subprocess.call(["./rawLogsGenScript.sh", caseStart, caseEnd, "0"])

    log_file_list = [
        "../../OUTPUT_PERFORMANCE_LOGS_HIP_NEW/Tensor_hip_audio_raw_performance_log.txt"
        ]

    functionality_group_list = [
        "to_decibels",
        "pre_emphasis_filter",
        "spectrogram"
    ]

    for log_file in log_file_list:

        # Opening log file
        try:
            f = open(log_file,"r")
            print("\n\n\nOpened log file -> ", log_file)
        except IOError:
            print("Skipping file -> ", log_file)
            continue

        stats = []
        maxVals = []
        minVals = []
        avgVals = []
        functions = []
        frames = []
        prevLine = ""
        funcCount = 0

        # Loop over each line
        for line in f:
            for functionality_group in functionality_group_list:
                if functionality_group in line:
                    # print(line)
                    # print(functionality_group)
                    functions.extend([" ", functionality_group, " "])
                    frames.extend([" ", " ", " "])
                    maxVals.extend([" ", " ", " "])
                    minVals.extend([" ", " ", " "])
                    avgVals.extend([" ", " ", " "])

            if "max,min,avg" in line:
                split_word_start = "Running "
                split_word_end = " 100"

                prevLine = (prevLine.partition(split_word_start)[2].partition(split_word_end)[0])
                if prevLine in functions:
                    functions.append(prevLine)
                    frames.append("100")
                    split_word_start = "max,min,avg = "
                    split_word_end = "\n"
                    stats = line.partition(split_word_start)[2].partition(split_word_end)[0].split(",")
                    maxVals.append(stats[0])
                    minVals.append(stats[1])
                    avgVals.append(stats[2])
                    funcCount += 1

            if line != "\n":
                prevLine = line

        # Print log lengths
        print("Functionalities - ", funcCount)

        # Print summary of log
        print("\n\nFunctionality\t\t\t\t\t\tFrames Count\tmax(s)\t\tmin(s)\t\tavg(s)\n")
        if len(functions) != 0:
            maxCharLength = len(max(functions, key=len))
            functions = [x + (' ' * (maxCharLength - len(x))) for x in functions]
            for i, func in enumerate(functions):
                print(func, "\t", frames[i], "\t\t", maxVals[i], "\t", minVals[i], "\t", avgVals[i])
        else:
            print("No variants under this category")

        # Closing log file
        f.close()

elif profilingOption == "YES":
    subprocess.call(["./rawLogsGenScript.sh", caseStart, caseEnd, "1"])

    RESULTS_DIR = "../../OUTPUT_PERFORMANCE_LOGS_HIP_NEW"
    print("RESULTS_DIR = " + RESULTS_DIR)
    CONSOLIDATED_FILE = RESULTS_DIR + "/consolidated_results.stats.csv"

    CASE_NUM_LIST = range(int(caseStart), int(caseEnd) + 1, 1)
    BIT_DEPTH_LIST = [2]

    # Open csv file
    new_file = open(RESULTS_DIR + "/consolidated_results.stats.csv",'w')
    new_file.write('"HIP Kernel Name","Calls","TotalDurationNs","AverageNs","Percentage"\n')
    prev=""

    # Loop through cases
    for CASE_NUM in CASE_NUM_LIST:
        # Set results directory
        CASE_RESULTS_DIR = RESULTS_DIR
        print("CASE_RESULTS_DIR = " + CASE_RESULTS_DIR)

        # Loop through bit depths
        for BIT_DEPTH in BIT_DEPTH_LIST:
            CASE_FILE_PATH = CASE_RESULTS_DIR + "/output_case" + str(CASE_NUM) + "_bitDepth" + str(BIT_DEPTH) + ".stats.csv"
            print("CASE_FILE_PATH = " + CASE_FILE_PATH)
            try:
                case_file = open(CASE_FILE_PATH,'r')
                for line in case_file:
                    print(line)
                    if not(line.startswith('"Name"')):
                        new_file.write(line)
                case_file.close()
            except IOError:
                print("Unable to open case results")
                continue

            new_file.close()
            os.system('chown $USER:$USER ' + RESULTS_DIR + "/consolidated_results.stats.csv")
    try:
        import pandas as pd
        pd.options.display.max_rows = None

        # Generate performance report
        df = pd.read_csv(RESULTS_DIR + "/consolidated_results.stats.csv")
        df["AverageMs"] = df["AverageNs"] / 1000000
        dfPrint = df.drop(['Percentage'], axis=1)
        dfPrint["HIP Kernel Name"] = dfPrint.iloc[:,0].str.lstrip("Hip_")
        dfPrint_noIndices = dfPrint.astype(str)
        dfPrint_noIndices.replace(['0', '0.0'], '', inplace=True)
        dfPrint_noIndices = dfPrint_noIndices.to_string(index=False)
        print(dfPrint_noIndices)

    except ImportError:
        print("\nPandas not available! Results of GPU profiling experiment are available in the following files:\n" + \
            CONSOLIDATED_FILE + "\n")
    except IOError:
        print("Unable to open results in " + RESULTS_DIR + "/consolidated_results.stats.csv")