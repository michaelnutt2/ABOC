
import os
import time
import argparse
import numpy as np
import torch
import shutil
import tempfile

from Preparedata.data import dataPrepare
from networkTool import device, reload
from octAttention import model
import encoderTool
import decoder
import pt as pointCloud

def get_file_size(path):
    return os.path.getsize(path)

def calculate_fidelity(original_ply, reconstructed_ply):
    # Use pt.pcerror if available, or simple distance check
    # Trying to reuse existing tools
    print("Calculating fidelity...")
    try:
        # Assuming pt.pcerror is a wrapper around the MPEG pcc_error_metric or similar
        # If it's an async subprocess, wait for it
        # pt.pcerror(file1, file2, normal, resolution, bounding_box)

        # Check if 'pc_error' command exists or use Python based check
        # For now, let's load points and do a quick cKDTree check if the library is missing
        # But 'pt' module seems to have it.

        # We need to load PLY to arrays first for some 'pt' functions,
        # but 'pcerror' likely takes file paths or arrays.
        # decoder.py lines 407: pt.pcerror(p, DQpt, None, '-r 1', None).wait()

        # We need p (N,3) and DQpt (N,3).
        # Let's load them using pointCloud.ptread

        p = pointCloud.ptread(original_ply)
        q = pointCloud.ptread(reconstructed_ply)

        # This returns numpy arrays

        # Calculate Peak Signal to Noise Ratio (PSNR) approx?
        # Or just return geometric error.

        proc = pointCloud.pcerror(p, q, None, '-r 1023', None) # Resolution 1023 is standard for 10-bit
        # It initiates a process.
        proc.wait()
        # The output is printed to stdout usually.

        # Alternatively, calculate MSE manually
        # This is expensive for large clouds in Python, but let's try strict Euclidean.
        # Skipping pure python implementation to trust 'pcerror' output.

    except Exception as e:
        print(f"Error calculating fidelity using pt.pcerror: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test ABOC Compression Pipeline")
    parser.add_argument("--file", type=str, required=True, help="Input Point Cloud File (PLY/BIN)")
    parser.add_argument("--model", type=str, default="modelsave/lidar/encoder_epoch_00801460.pth", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="test_output", help="Directory for artifacts")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=args.output_dir)
    print(f"Using temp directory: {temp_dir}")

    input_basename = os.path.splitext(os.path.basename(args.file))[0]

    try:
        # 1. Load Model
        print(f"Loading model from {args.model}...")
        loaded_model = model.to(device)
        saveDic = reload(None, args.model, multiGPU=False)
        loaded_model.load_state_dict(saveDic['encoder'])
        loaded_model.eval()

        # 2. Prepare Data (Encode Phase 1)
        print("Preparing data (Generating FlatBuffer)...")
        t0 = time.time()
        # dataPrepare(fileName, saveMatDir, offset, qs, rotation, normalize)
        # We use standard params from 'encoder.py'
        # qs=2 / (2 ** 12 - 1) -> This seems specific to a certain scale?
        # Let's use 1.0 or whatever 'encoder.py' used: 2 / (2 ** 12 - 1)
        # encoder.py line 39: qs=2 / (2 ** 12 - 1), rotation=False, normalize=True

        fb_file, DQpt_verify, normalizePt = dataPrepare(
            args.file,
            saveMatDir=temp_dir,
            offset='min',
            qs=2 / (2 ** 12 - 1),
            rotation=False,
            normalize=True
        )
        t_prepare = time.time() - t0
        print(f"Data preparation took {t_prepare:.2f}s")

        # Load FlatBuffer to dict for encoderTool
        import OctreeData.Dataset as Dataset
        with open(fb_file, 'rb') as f:
            buf = f.read()
            dataset = Dataset.Dataset.GetRootAsDataset(buf, 0)

        nodes_len = dataset.NodesLength()
        features = np.zeros((nodes_len, 17, 3), dtype=np.int32)
        targets = np.zeros(nodes_len, dtype=np.int32)

        # Extract Loop (Simplified from encoder.py)
        for i in range(nodes_len):
            node = dataset.Nodes(i)
            parent_occ = node.ParentOccupancy()
            neighbors_len = node.NeighborOccupanciesLength()
            neighbors = np.zeros(neighbors_len, dtype=int)
            for j in range(neighbors_len):
                neighbors[j] = node.NeighborOccupancies(j)

            full_ctx = np.zeros(17, dtype=int)
            full_ctx[0:8] = neighbors[0:8]
            full_ctx[8] = parent_occ
            full_ctx[9:] = neighbors[8:]

            features[i, :, 0] = full_ctx
            features[i, :, 1] = node.Level() - 1
            features[i, :, 2] = node.Octant()
            targets[i] = node.Occupancy()

        data_packet = {
            'features': features,
            'targets': targets,
            'ptName': input_basename,
            'visualize_data': (normalizePt, DQpt_verify)
        }

        # 3. Compress (Encode Phase 2)
        bin_file = os.path.join(temp_dir, f"{input_basename}.bin")
        print("Encoding...")
        bpp, t_encode, compressed_size = encoderTool.compress(data_packet, loaded_model, outputfile=bin_file, actualcode=True)

        original_size = get_file_size(args.file)
        ratio = original_size / compressed_size if compressed_size > 0 else 0

        # 4. Decode
        print("Decoding...")
        # decodeOct(binfile, oct_data_seq, model, bptt)
        # Note: 'decoder.py' signature: def decodeOct(binfile, oct_data_seq, model, bptt)
        # 'oct_data_seq' is only used for checking? Or needed?
        # It uses len(oct_data_seq) for initialization?
        # Let's pass targets separately or handle it.
        # decoder.py seems to require 'oct_data_seq' to know length?
        # "oct_len = len(oct_data_seq)" -> Yes.
        # In a real scenario we'd header this, but here we can pass it.

        t1 = time.time()
        # Note: bptt is just a constant usually from 'networkTool'
        decoded_points = decoder.decodeOct(bin_file, targets, loaded_model, 0) # bptt unused in loop? Check decoder.py
        t_decode = time.time() - t1
        print(f"Decoding took {t_decode:.2f}s")

        # 5. Reconstruct
        # Dequantization
        # We need 'qs' and 'offset' from the dataset metadata
        qs_val = dataset.Qs()
        offset_val = np.array([dataset.OffsetX(), dataset.OffsetY(), dataset.OffsetZ()])

        # decoded_points comes from 'DeOctree', which returns quantized integers?
        # Yes.

        rec_points = (decoded_points * qs_val + offset_val)

        rec_ply = os.path.join(args.output_dir, f"{input_basename}_rec.ply")
        pointCloud.write_ply_data(rec_ply, rec_points)
        print(f"Saved reconstructed PLY to {rec_ply}")

        # 6. Report
        print("\n" + "="*50)
        print("COMPRESSION REPORT")
        print("="*50)
        print(f"Original File: {args.file}")
        print(f"Original Size: {original_size / 1024:.2f} KB")
        print(f"Compressed Size: {compressed_size / 1024:.2f} KB")
        print(f"Compression Ratio: {ratio:.2f}x")
        print(f"Bits Per Point (BPP): {bpp:.4f}")
        print("-"*30)
        print(f"Encoding Time:   {t_encode:.4f}s")
        print(f"Decoding Time:   {t_decode:.4f}s")
        print(f"Total Latency:   {t_encode + t_decode:.4f}s")
        print("-"*30)

        calculate_fidelity(args.file, rec_ply)

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup temp dir
        # shutil.rmtree(temp_dir)
        print(f"Temporary files left in {temp_dir} for inspection")

if __name__ == "__main__":
    main()
