
import os
import sys
import time
import glob
import random
import csv
import numpy as np
import tqdm

from Preparedata.data import dataPrepare
import numpyAc
# We only need Dataset reading logic, not the TF model
import OctreeData.Dataset as Dataset
import OctreeData.OctreeNode as OctreeNode

# Configuration
SEQUENCES_DIR = "/home/michael-nutt/Datasets/SemanticKITTI/dataset/sequences"
OUTPUT_DIR = "./Exp/Ablation_Frequency"
LOG_FILE = f"{OUTPUT_DIR}/ablation_log.txt"
CSV_FILE = f"{OUTPUT_DIR}/ablation_results.csv"
SAMPLES_PER_SEQUENCE = 100 # Same sampling as benchmark for fair comparison

def print_log(*args):
    msg = " ".join(map(str, args))
    print(msg)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + "\n")


from Octree import dec2bin # Needed for decoding logic

def process_file(ori_file):
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(ori_file))[0]

    # 1. Prepare Data
    if ori_file.endswith('.fb'):
        fb_file = ori_file
        normalize_pt_len = 0
    else:
        fb_file, _, normalize_pt = dataPrepare(
            ori_file,
            saveMatDir=f'{OUTPUT_DIR}/temp_data',
            offset='min',
            qs=2 / (2 ** 12 - 1),
            rotation=False,
            normalize=True
        )
        normalize_pt_len = len(normalize_pt)

    # 2. Load FlatBuffer to get Targets
    with open(fb_file, 'rb') as f:
        buf = f.read()
        dataset = Dataset.Dataset.GetRootAsDataset(buf, 0)

    nodes_len = dataset.NodesLength()
    targets = np.zeros(nodes_len, dtype=np.int32)

    for i in range(nodes_len):
        node = dataset.Nodes(i)
        targets[i] = node.Occupancy()

    # 3. Calculate Frequencies
    counts = np.bincount(targets, minlength=256)
    total_counts = np.sum(counts)
    probs = counts.astype(np.float32) / total_counts
    probs += 1e-9
    probs /= probs.sum()

    # Broadcast for Encoding
    probs_all = np.tile(probs, (nodes_len, 1))

    # 4. Encode
    output_bin = f"{OUTPUT_DIR}/bins/{base_name}.bin"
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)

    ac = numpyAc.arithmeticCoding()
    targets_int = targets.astype(np.int16)

    byte_stream, _ = ac.encode(probs_all, targets_int, binfile=output_bin)
    file_size = len(byte_stream)
    enc_duration = time.time() - start_time

    # 5. Decode (New)
    dec_start = time.time()

    # In a real scenario, 'probs' would be read from header. Here we reuse it.
    # Initialize Decoder
    estimated_len = file_size * 2
    ac_dec = numpyAc.arithmeticDeCoding(None, estimated_len, 255, output_bin)

    # Root
    probs_expanded = np.expand_dims(probs, 0)
    try:
        root_occ = ac_dec.decode(probs_expanded)
    except Exception:
        root_occ = 1 # Fail safe

    all_octs = [root_occ]
    current_level_indices = [0]
    current_level = 0
    MAX_LEVEL = 21 # Matching benchmark depth

    while current_level_indices:
        if current_level >= MAX_LEVEL:
            break

        # Determine how many children to decode
        batch_size = 0
        for p_idx in current_level_indices:
            p_occ = all_octs[p_idx]
            bits = dec2bin(p_occ, 8)
            child_existence = bits[::-1]
            batch_size += sum(child_existence)

        if batch_size == 0:
            break

        # Prepare probabilities (All same)
        # We need [batch_size, 256]
        probs_batch = np.tile(probs, (batch_size, 1))

        # Decode Batch
        # Note: numpyAc.decode processes one by one internally if we loop,
        # or we might need to loop here since we don't have a 'bulk decode' wrapper easily exposed
        # that takes variable length inputs without rebuilding the C++ object?
        # decoder_tf.py loops: `child_occ = ac.decode(prob_dist)`

        next_level_indices = []

        # We can just loop and decode efficiently because we don't need Model Inference!
        # The bottleneck in decoder_tf is Model Inference. Here it's pure arithmetic decoding.

        for _ in range(batch_size):
             child_occ = ac_dec.decode(probs_expanded) # probs_expanded is [1, 256]
             all_octs.append(child_occ)
             next_level_indices.append(len(all_octs) - 1)

        current_level_indices = next_level_indices
        current_level += 1

    dec_duration = time.time() - dec_start

    # 6. Metrics
    if normalize_pt_len > 0:
        bpp = (file_size * 8) / normalize_pt_len
    else:
        bpp = 0.0

    return bpp, file_size, enc_duration, dec_duration, normalize_pt_len

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/temp_data", exist_ok=True)

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence", "File", "NumPoints", "EncTime", "DecTime", "CompressedSize", "BPP"])

    print_log("Starting Frequency Ablation Study (Enc+Dec)...")

    sequences = sorted(glob.glob(f"{SEQUENCES_DIR}/*"))

    total_files = 0
    total_size = 0
    total_points = 0
    total_enc_time = 0
    total_dec_time = 0

    for seq_path in sequences:
        seq_id = os.path.basename(seq_path)
        velodyne_dir = os.path.join(seq_path, "velodyne")

        if not os.path.exists(velodyne_dir):
            continue

        all_bins = glob.glob(f"{velodyne_dir}/*.bin")
        if not all_bins:
            continue

        if len(all_bins) > SAMPLES_PER_SEQUENCE:
            sampled_files = random.sample(all_bins, SAMPLES_PER_SEQUENCE)
        else:
            sampled_files = all_bins

        print_log(f"Processing Sequence {seq_id}: {len(sampled_files)} files")

        for ori_file in sampled_files:
            file_name = os.path.basename(ori_file)
            try:
                bpp, comp_size, enc_time, dec_time, num_points = process_file(ori_file)

                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([seq_id, file_name, num_points, f"{enc_time:.4f}", f"{dec_time:.4f}", comp_size, f"{bpp:.4f}"])

                total_files += 1
                total_size += comp_size
                total_points += num_points
                total_enc_time += enc_time
                total_dec_time += dec_time

            except Exception as e:
                print_log(f"Error {file_name}: {e}")
                import traceback
                traceback.print_exc()

    if total_files > 0:
        avg_bpp = (total_size * 8) / total_points
        avg_enc = total_enc_time / total_files
        avg_dec = total_dec_time / total_files

        print_log("-" * 30)
        print_log(f"Ablation Complete.")
        print_log(f"Total Files: {total_files}")
        print_log(f"Avg Enc Time: {avg_enc:.4f}s")
        print_log(f"Avg Dec Time: {avg_dec:.4f}s")
        print_log(f"Avg BPP: {avg_bpp:.4f}")

if __name__ == "__main__":
    main()
