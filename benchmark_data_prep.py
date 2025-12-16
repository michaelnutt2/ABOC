import os
import time
import glob
import argparse
import random
import numpy as np
from Preparedata.data import dataPrepare

def benchmark_data_prep(data_dir, num_samples=100, output_dir='./Data/benchmark_tmp'):
    # 1. Gather all .bin files
    search_pattern = os.path.join(data_dir, "**/*.bin")
    print(f"Searching for raw Lidar files in: {search_pattern}")
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        print("No .bin files found. Please check the data directory.")
        return

    print(f"Found {len(all_files)} files.")

    # 2. Sample
    num_to_test = min(len(all_files), num_samples)
    samples = random.sample(all_files, num_to_test)
    print(f"Benchmarking {num_to_test} random samples...")

    # 3. Benchmark Loop
    times = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        for idx, f in enumerate(samples):
            # Run dataPrepare
            t0 = time.time()

            # signature: dataPrepare(fileName, saveMatDir='Data', ...)
            # dataPrepare returns (fb_path, dq_pt, normalize_pt)
            try:
                # We save to a tmp dir to avoid cluttering real data dirs
                dataPrepare(f, saveMatDir=output_dir, offset='min', qs=2/(2**12-1), rotation=False, normalize=True)
            except Exception as e:
                print(f"Error processing {f}: {e}")
                continue

            duration = time.time() - t0
            times.append(duration)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx+1}/{num_to_test} - Last time: {duration:.4f}s")

    finally:
        # Cleanup Benchmark Outputs
        print("Cleaning up temporary files...")
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)

    # 4. Report
    if times:
        avg_time = np.mean(times)
        hz = 1.0 / avg_time
        print("\n==========================================")
        print("       DATA PREPARATION BENCHMARK         ")
        print("==========================================")
        print(f"Files Processed: {len(times)}")
        print(f"Total Time:      {sum(times):.4f} s")
        print(f"Average Time:    {avg_time:.4f} s")
        print(f"Throughput:      {hz:.2f} Hz")
        print("==========================================")
    else:
        print("No files successfully processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Data Preparation Step (CPU)")
    parser.add_argument('--data_dir', type=str, default='/home/michael-nutt/Datasets/SemanticKITTI/dataset/sequences/', help="Directory containing raw .bin files")
    parser.add_argument('--samples', type=int, default=100, help="Number of files to sample")

    args = parser.parse_args()

    benchmark_data_prep(args.data_dir, args.samples)
