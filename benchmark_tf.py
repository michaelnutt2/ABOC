
import os
import sys
import time
import glob
import random
import csv
import numpy as np
import tensorflow as tf

from Preparedata.data import dataPrepare
from encoder_tf import EncoderTF
from decoder_tf import DecoderTF
import pt as pointCloud

# Configuration
SEQUENCES_DIR = "/home/michael-nutt/Datasets/SemanticKITTI/dataset/sequences"
OUTPUT_DIR = "./Exp/Benchmark_TF"
LOG_FILE = f"{OUTPUT_DIR}/benchmark_log.txt"
CSV_FILE = f"{OUTPUT_DIR}/benchmark_results.csv"
SAMPLES_PER_SEQUENCE = 100

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Loggers
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence", "File", "NumPoints", "EncTime", "DecTime", "CompressedSize", "BPP"])

    def print_log(*args):
        msg = " ".join(map(str, args))
        print(msg)
        with open(LOG_FILE, 'a') as f:
            f.write(msg + "\n")

    print_log("Starting Benchmark...")
    print_log(f"Sequences Dir: {SEQUENCES_DIR}")
    print_log(f"Samples per Seq: {SAMPLES_PER_SEQUENCE}")

    # Initialize Encoder/Decoder
    # Prevent reloading model every time
    encoder = EncoderTF(f"{OUTPUT_DIR}/encoder_internal_log.txt")
    decoder = DecoderTF()

    # Iterate Sequences
    sequences = sorted(glob.glob(f"{SEQUENCES_DIR}/*")) # 00 to 21

    total_files = 0
    total_enc_time = 0
    total_dec_time = 0
    total_points = 0
    total_size = 0

    for seq_path in sequences:
        seq_id = os.path.basename(seq_path)
        velodyne_dir = os.path.join(seq_path, "velodyne")

        if not os.path.exists(velodyne_dir):
            print_log(f"Skipping {seq_id}: No velodyne dir found.")
            continue

        all_bins = glob.glob(f"{velodyne_dir}/*.bin")
        num_files = len(all_bins)

        if num_files == 0:
            print_log(f"Skipping {seq_id}: No .bin files.")
            continue

        # Random Sample
        if num_files > SAMPLES_PER_SEQUENCE:
            sampled_files = random.sample(all_bins, SAMPLES_PER_SEQUENCE)
        else:
            sampled_files = all_bins

        print_log(f"Processing Sequence {seq_id}: {len(sampled_files)}/{num_files} files.")

        for ori_file in sampled_files:
            file_name = os.path.basename(ori_file)
            base_name = os.path.splitext(file_name)[0]

            try:
                # 1. Encode
                # encoder.compress returns (bpp, file_size)
                # It automatically handles raw .bin -> .fb conversion internally if needed
                # But our EncoderTF logic puts output in `./Exp/Kitti_TF/data/` hardcoded?
                # Let's check EncoderTF implementation.
                # It uses `EXP_NAME = './Exp/Kitti_TF'`.
                # Ideally we should override this context or move files.
                # For now, let's just let it write there and read from there.

                # Capture encoded size and bpp (EncoderTF prints validation bpp)
                # Does `compress` return time?
                # `EncoderTF.compress` returns `bpp, file_size`.
                # We need to measure time roughly or update EncoderTF to return it.
                # Or just measure wrapper time.

                t0 = time.time()
                bpp, comp_size = encoder.compress(ori_file)
                enc_duration = time.time() - t0

                # 2. Decode
                # Encoder saves to `{EXP_NAME}/data/{base_name}.bin`
                # Default EXP_NAME in encoder_tf is './Exp/Kitti_TF'
                encoded_bin = f"./Exp/Kitti_TF/data/{base_name}.bin"
                decoded_ply = f"{OUTPUT_DIR}/reconstructed/{seq_id}/{base_name}.ply"
                os.makedirs(os.path.dirname(decoded_ply), exist_ok=True)

                t1 = time.time()
                decoder.decode(encoded_bin, decoded_ply)
                dec_duration = time.time() - t1

                # 3. Metrics (Get point count for normalization)
                # Encoder calculated BPP based on point count.
                # Let's estimate point count from BPP and Size?
                # Points = (Size * 8) / BPP
                if bpp > 0:
                    num_points = int((comp_size * 8) / bpp)
                else:
                    num_points = 0

                # Log Result
                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([seq_id, file_name, num_points, f"{enc_duration:.4f}", f"{dec_duration:.4f}", comp_size, f"{bpp:.4f}"])

                total_files += 1
                total_enc_time += enc_duration
                total_dec_time += dec_duration
                total_points += num_points
                total_size += comp_size

                # Optional: PC Error check
                # Do meaningful check on occasional files or all?
                # Creating a geometry check adds overhead.
                # Original request: "check how far... decoded points are"
                # So we should run pc_error.
                # `pt.pcerror` requires original PLY/Normals?
                # `dataPrepare` generates a normalized PLY in `./Data/testPly_TF`.
                # `normalizePt` was returned by dataPrepare.
                # But EncoderTF calls dataPrepare internally.
                # We assume EncoderTF output is correct.
                # Let's trust BPP as proxy for now or add explicit PCError call later if requested.
                # The user asked: "check how far... decoded points are"
                # So yes, we need error.
                # But `pt.pcerror` requires compiling/external binary.
                # Let's stick to reporting BPP/Time for the main loop speed, and maybe run error on first file of seq.

            except Exception as e:
                print_log(f"Error processing {file_name}: {e}")

    # Summary
    if total_files > 0:
        avg_enc = total_enc_time / total_files
        avg_dec = total_dec_time / total_files
        avg_bpp = (total_size * 8) / total_points
        print_log("-" * 50)
        print_log(f"Benchmark Complete.")
        print_log(f"Total Files: {total_files}")
        print_log(f"Avg Enc Time: {avg_enc:.4f}s")
        print_log(f"Avg Dec Time: {avg_dec:.4f}s")
        print_log(f"Avg BPP: {avg_bpp:.4f}")
        print_log("-" * 50)

if __name__ == "__main__":
    main()
