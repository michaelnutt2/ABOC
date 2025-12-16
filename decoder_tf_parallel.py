
import os
import sys
import time
import numpy as np
import tensorflow as tf
from collections import deque

import pt as pointCloud
from Octree import DeOctree, dec2bin
import numpyAc
from tf_model import create_model
from config import CONTEXT_LEN, MAX_OCTREE_LEVEL, EXP_NAME

# Configuration
# EXP_NAME is now in config
MODEL_WEIGHTS_PATH = 'modelsave/lidar/checkpoints_model_epoch_50.weights.h5'
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

class DecoderTFParallel:
    def __init__(self, model_path=MODEL_WEIGHTS_PATH):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        print(f"Loading TensorFlow model on {DEVICE}...")
        model = create_model()
        dummy_input = tf.zeros((1, CONTEXT_LEN, 3), dtype=tf.int32)
        model(dummy_input)

        if os.path.exists(self.model_path):
            print(f"Loading weights from {self.model_path}")
            model.load_weights(self.model_path)
        else:
            print(f"Warning: Weights not found at {MODEL_WEIGHTS_PATH}. Decoding will be random.")

        return model

    def decode(self, bin_file, output_ply):
        start_time = time.time()
        print(f"Parallel Decoding {bin_file}...")

        # 1. Initialize Arithmetic Decoder
        estimated_len = os.path.getsize(bin_file) * 2
        ac = numpyAc.arithmeticDeCoding(None, estimated_len, 255, bin_file)

        # 2. Decode Root
        init_context = tf.zeros((1, CONTEXT_LEN, 3), dtype=tf.int32)
        logits = self.model(init_context, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0] # [256]
        probs_expanded = np.expand_dims(probs, 0)

        try:
            root_occ = ac.decode(probs_expanded)
        except Exception as e:
            print(f"Decoding Error at Root: {e}")
            return

        all_octs = [root_occ]
        current_level_indices = [0]
        current_level = 0

        while current_level_indices:
            batch_features = []

            # Step A: Prepare batch inputs (Find all children)
            # This part is still Python loop but it's simple integer logic (fast enough compared to Model/AC)
            for p_idx in current_level_indices:
                p_occ = all_octs[p_idx]
                bits = dec2bin(p_occ, 8)
                child_existence = bits[::-1]

                # Context Reconstruction
                # Dynamic Logic based on CONTEXT_LEN
                mid = CONTEXT_LEN // 2
                radius = mid # e.g. 512 for len 1025

                ctx_start = max(0, p_idx - radius)
                ctx_end = min(len(all_octs), p_idx + radius + 1)

                full_ctx = np.zeros(CONTEXT_LEN, dtype=int)
                for k in range(ctx_start, ctx_end):
                    target_idx = mid + (k - p_idx)
                    full_ctx[target_idx] = all_octs[k]

                for octant in range(8):
                    if child_existence[octant] == 1:
                        feat = np.zeros((CONTEXT_LEN, 3), dtype=np.int32)
                        feat[:, 0] = full_ctx
                        feat[:, 1] = current_level
                        feat[mid, 2] = octant
                        batch_features.append(feat)

            if not batch_features:
                break

            # Step B: Batch Inference
            batch_arr = np.array(batch_features)

            # Use large batch size for inference efficiency
            ds = tf.data.Dataset.from_tensor_slices(batch_arr).batch(512)

            all_probs = []
            for b_feat in ds:
                logits = self.model(b_feat, training=False)
                probs = tf.nn.softmax(logits, axis=-1).numpy()
                all_probs.append(probs)

            all_probs = np.concatenate(all_probs, axis=0) # [TotalChildren, 256]

            # Step C: Parallel (Batch) Arithmetic Decoding (OPTIMIZED)
            try:
                # Call the C++ batch decoder directly
                child_occs = ac.decode_batch(all_probs)

                # Append results
                all_octs.extend(child_occs)

                # Calculate indices for next level
                start_index = len(all_octs) - len(child_occs)
                next_level_indices = list(range(start_index, len(all_octs)))

            except Exception as e:
                print(f"Batch Decoding Error: {e}")
                import traceback
                traceback.print_exc()
                break

            current_level_indices = next_level_indices
            current_level += 1
            print(f"Decoded Level {current_level}, Total Nodes: {len(all_octs)}")

            if current_level >= MAX_OCTREE_LEVEL:
                print(f"Reached Max Level {MAX_OCTREE_LEVEL}. Stopping.")
                break

        duration = time.time() - start_time
        print(f"Parallel decoding finished in {duration:.2f}s. Total Nodes: {len(all_octs)}")

        # 3. Reconstruct
        try:
            rec_pts = DeOctree(all_octs)
            pointCloud.write_ply_data(output_ply, rec_pts)
            print(f"Saved PLY to {output_ply}")
        except Exception as e:
            print(f"Reconstruction Error: {e}")

if __name__ == "__main__":
    decoder = DecoderTFParallel()

    # Matches the one from encoder_tf.py
    bin_file = f"{EXP_NAME}/data/Kitti_00002510.bin"
    output_ply = f"{EXP_NAME}/test/Kitti_00002510_parallel_rec.ply"

    os.makedirs(os.path.dirname(output_ply), exist_ok=True)

    if os.path.exists(bin_file):
        decoder.decode(bin_file, output_ply)
    else:
        print(f"Bin file not found: {bin_file}")
