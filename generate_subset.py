import os
import glob
import random
import multiprocessing
import config
from Preparedata.data import dataPrepare

def process_file_wrapper(args):
    """Wrapper to handle exceptions and logging in worker process."""
    file_path, output_dir = args
    try:
        # dataPrepare(fileName, saveMatDir=..., ...)
        # It generates output file name based on input + dir
        # We assume dataset structure (sequences/XX/velodyne/YYYY.bin)

        # Extract metadata for naming if needed, but dataPrepare handles it intuitively usually
        # We pass minimal args for defaults
        dataPrepare(
            file_path,
            saveMatDir=output_dir,
            offset='min',
            qs=2 / (2 ** 12 - 1), # 12-bit quantization
            normalize=True
        )
        return None
    except Exception as e:
        return f"Error {os.path.basename(file_path)}: {e}"

def main():
    # 1. Configuration
    NUM_SAMPLES = 500
    OUTPUT_DIR = "Data/subset_64"
    DATA_ROOT = "/home/michael-nutt/Datasets/SemanticKITTI/dataset/sequences"

    print(f"--- Generating {NUM_SAMPLES} Random Samples ---")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Context Range (from config): {config.CONTEXT_RANGE}")

    if os.path.exists(OUTPUT_DIR):
        print(f"Warning: Output directory {OUTPUT_DIR} already exists.")
    else:
        os.makedirs(OUTPUT_DIR)

    # 2. Find Files
    search_pattern = os.path.join(DATA_ROOT, "*/velodyne/*.bin")
    print(f"Scanning {search_pattern}...")
    all_files = glob.glob(search_pattern)

    if not all_files:
        print("No .bin files found!")
        return

    print(f"Found {len(all_files)} total files.")

    # 3. Random Sample
    if len(all_files) > NUM_SAMPLES:
        selected_files = random.sample(all_files, NUM_SAMPLES)
    else:
        selected_files = all_files

    print(f"Selected {len(selected_files)} files for processing.")

    # 4. Multiprocessing
    tasks = [(f, OUTPUT_DIR) for f in selected_files]

    cpu_count = multiprocessing.cpu_count()
    print(f"Processing with {cpu_count} cores...")

    with multiprocessing.Pool(cpu_count) as pool:
        # Use imap for progress reporting
        for i, res in enumerate(pool.imap_unordered(process_file_wrapper, tasks)):
            if res:
                print(res) # Print errors

            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(tasks)}")

    print("Done! You can now upload 'Data/subset_64' to GCS.")

if __name__ == "__main__":
    main()
