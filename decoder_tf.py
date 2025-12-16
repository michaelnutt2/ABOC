
import os
import sys
import time
import numpy as np
import tensorflow as tf
from collections import deque

import pt as pointCloud
from Octree import DeOctree, dec2bin
import numpyAc
from tf_model import create_model, CONTEXT_LEN, MAX_OCTREE_LEVEL

# Configuration
EXP_NAME = './Exp/Kitti_TF'
MODEL_WEIGHTS_PATH = 'modelsave/lidar/checkpoints_model_epoch_50.weights.h5'
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

class DecoderTF:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading TensorFlow model on {DEVICE}...")
        model = create_model()
        dummy_input = tf.zeros((1, CONTEXT_LEN, 3), dtype=tf.int32)
        model(dummy_input)

        if os.path.exists(MODEL_WEIGHTS_PATH):
            print(f"Loading weights from {MODEL_WEIGHTS_PATH}")
            model.load_weights(MODEL_WEIGHTS_PATH)
        else:
            print(f"Warning: Weights not found at {MODEL_WEIGHTS_PATH}. Decoding will be random.")

        return model

    def decode(self, bin_file, output_ply):
        start_time = time.time()
        print(f"Decoding {bin_file}...")

        # 1. Initialize Arithmetic Decoder
        # NOTE: Using a fixed large length for now or file-size based estimate.
        estimated_len = os.path.getsize(bin_file) * 2 # Crude estimate
        ac = numpyAc.arithmeticDeCoding(None, estimated_len, 255, bin_file)

        # 2. Sequential Decoding
        # Phase 1: Decode Root
        init_context = tf.zeros((1, CONTEXT_LEN, 3), dtype=tf.int32)
        logits = self.model(init_context, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0] # [256]

        probs_expanded = np.expand_dims(probs, 0) # [1, 256]

        try:
            root_occ = ac.decode(probs_expanded)
        except Exception as e:
            print(f"Decoding Error at Root: {e}")
            return

        all_octs = [root_occ]
        current_level_indices = [0] # Index in all_octs
        current_level = 0

        while current_level_indices:
            batch_features = []
            batch_meta = [] # (parent_idx_in_all_octs, child_octant)

            # Step A: Prepare batch inputs (Find all children)
            for p_idx in current_level_indices:
                p_occ = all_octs[p_idx]

                # Identify existing children
                # dec2bin(val, 8) -> [MSB...LSB]
                bits = dec2bin(p_occ, 8)
                child_existence = bits[::-1] # [Octant0 ... Octant7]

                # Context (from all_octs)
                ctx_start = max(0, p_idx - 8)
                ctx_end = min(len(all_octs), p_idx + 9)
                full_ctx = np.zeros(CONTEXT_LEN, dtype=int)

                for k in range(ctx_start, ctx_end):
                    target_idx = 8 + (k - p_idx)
                    full_ctx[target_idx] = all_octs[k]

                for octant in range(8):
                    if child_existence[octant] == 1:
                        # Prepare input feature
                        feat = np.zeros((CONTEXT_LEN, 3), dtype=np.int32)
                        feat[:, 0] = full_ctx
                        feat[:, 1] = current_level
                        feat[8, 2] = octant

                        batch_features.append(feat)
                        batch_meta.append((p_idx, octant))

            if not batch_features:
                break

            # Step B: Batch Inference
            batch_arr = np.array(batch_features)

            ds = tf.data.Dataset.from_tensor_slices(batch_arr).batch(512)

            all_probs = []
            for b_feat in ds:
                logits = self.model(b_feat, training=False)
                probs = tf.nn.softmax(logits, axis=-1).numpy()
                all_probs.append(probs)

            all_probs = np.concatenate(all_probs, axis=0) # [TotalChildren, 256]

            # Step C: Sequential Arithmetic Decoding
            next_level_indices = []

            for i in range(len(batch_features)):
                # Decode
                prob_dist = np.expand_dims(all_probs[i], 0)
                try:
                    child_occ = ac.decode(prob_dist)
                    # Adjust if ac.decode returns value + 1?
                    # encoder_tf wrote `targets` (1-255).
                    # `numpyAc.encode` takes values.
                    # `numpyAc.decode` returns values.
                    # Previous decoder.py did `ac.decode() + 1`?
                    # Let's check `decoder.py` again.
                    # `child_occ_code = ac.decode(np.expand_dims(probs_batch[b_idx], 0)) + 1`
                    # Yes, it added 1!
                    # `encoder_tf.py`: targets elements are 1..255.
                    # `numpyAc.encode` calls `Encoder.encode(prob, sym)`.
                    # If `sym` is 1..255.
                    # `ac.decode` returns symbol index.
                    # If I passed `full_probs` where index 1 corresponds to symbol 1 probability.
                    # And `ac.decode` returns the index of the probability chosen?
                    # If `ac.decode` returns 0-based index.
                    # If index 1 was the high prob one. It returns 1.
                    # So maybe +1 is needed IF `ac.decode` returns something else?
                    # Or maybe `decoder.py` used `proBit` which was size 255 (class 1..255).
                    # And `ac.decode` on size 255 probability vector returns 0..244.
                    # Then +1 -> 1..255.
                    # MY `full_probs` IS SIZE 256.
                    # So `ac.decode` should return 0..255.
                    # So I likely DON'T need +1 if I built the PDF with 256 entries.
                    # The original `decoder.py` was likely building a PDF of size 255?
                    # `decoder.py`: `numpyAc.arithmeticDeCoding(..., 255, ...)`
                    # `decoder.py`: `proBit[i] = prob` (size 255).
                    # So `decoder.py` used size 255.
                    # I am using size 256.
                    # So `child_occ` should be correct as is.

                    pass
                except Exception as e:
                    print(f"Decoding Error: {e}")
                    child_occ = 1 # Fallback or break

                all_octs.append(child_occ)
                next_level_indices.append(len(all_octs) - 1)

            current_level_indices = next_level_indices
            current_level += 1
            print(f"Decoded Level {current_level}, Total Nodes: {len(all_octs)}")

            if current_level >= MAX_OCTREE_LEVEL:
                print(f"Reached Max Level {MAX_OCTREE_LEVEL}. Stopping decoding to prevent infinite loop.")
                break

        duration = time.time() - start_time
        print(f"Decoding finished in {duration:.2f}s. Total Nodes: {len(all_octs)}")

        # 3. Reconstruct Point Cloud
        try:
            rec_pts = DeOctree(all_octs)
            pointCloud.write_ply_data(output_ply, rec_pts)
            print(f"Saved PLY to {output_ply}")
        except Exception as e:
            print(f"Reconstruction Error: {e}")

if __name__ == "__main__":
    decoder = DecoderTF()

    # Matches the one from encoder_tf.py (Kitti_00002510.fb -> Kitti_00002510.bin)
    bin_file = f"{EXP_NAME}/data/Kitti_00002510.bin"
    output_ply = f"{EXP_NAME}/test/Kitti_00002510_rec.ply"

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)

    if os.path.exists(bin_file):
        decoder.decode(bin_file, output_ply)
    else:
        print(f"Bin file not found: {bin_file}")
