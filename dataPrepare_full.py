"""
Description: Prepare data for traning and testing (FULL DATASET RUN).
             Optimized with multiprocessing.
"""
import glob
from Preparedata.data import dataPrepare
from networkTool import CPrintl

import os
import multiprocessing
from functools import partial

def makedFile(dir):
    fileList = sorted(glob.glob(dir))
    return fileList

def process_file(args):
    file, outDir, ptNamePrefix, folder, qlevel = args
    fileName = folder + file.split('/')[-1][:-4]
    dataName = outDir + ptNamePrefix + fileName + '.fb'

    try:
        dataPrepare(file, saveMatDir=outDir, ptNamePrefix=ptNamePrefix + folder, offset='min',
                    qs=2 / (2 ** qlevel - 1), normalize=True)
        return dataName
    except Exception as e:
        return f"Error processing {file}: {e}"

if __name__ == "__main__":

    # ####For KITTI######
    oriDir = '/home/michael-nutt/Datasets/SemanticKITTI/dataset/sequences/'
    outDir = 'Data/Lidar/train/'
    ptNamePrefix = 'Kitti_'

    printl = CPrintl('Preparedata/makedFileLidar.log')
    makeFileList = set(makedFile(outDir + '*.fb')) # Use set for O(1) lookup



    tasks = []

    # FULL RUN CONFIGURATION
    # Process all sequences 00-21 without file limits
    for folder in range(0, 22):
        folder = '{:02d}'.format(folder)
        fileList = sorted(glob.glob(oriDir + folder + '/velodyne/*.bin'))

        for n, file in enumerate(fileList):
            # No limit break

            fileName = folder + file.split('/')[-1][:-4]
            dataName = outDir + 'Kitti_' + fileName + '.fb'

            if dataName in makeFileList:
                print(dataName, 'maked!')
                continue

            qlevel = 12
            # Append task arguments
            tasks.append((file, outDir, ptNamePrefix, folder, qlevel))

    print(f"Starting parallel processing of {len(tasks)} files with {multiprocessing.cpu_count()} cores...")

    if tasks:
        with multiprocessing.Pool() as pool:
            # Use imap_unordered for better responsiveness during processing
            for i, result in enumerate(pool.imap_unordered(process_file, tasks)):
                if (i + 1) % 100 == 0: # Log every 100 files for huge dataset
                    printl(result)

    print("Full Data preparation complete.")
