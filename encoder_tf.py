
import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
import tqdm

import pt as pointCloud
from Preparedata.data import dataPrepare
import numpyAc
from tf_model import create_model
from config import CONTEXT_LEN, MAX_OCTREE_LEVEL, EXP_NAME

# Check libraries
try:
    import OctreeData.Dataset as Dataset
    import OctreeData.OctreeNode as OctreeNode
except ImportError:
    print("Error: Could not import FlatBuffer classes. Run compile_schema.sh or ensure OctreeData is generated.")
    sys.exit(1)

# Configuration
# EXP_NAME is now in config
MODEL_WEIGHTS_PATH = 'modelsave/lidar/checkpoints_model_epoch_50.weights.h5'
DEVICE = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

class EncoderTF:
    def __init__(self, log_file, model_path=MODEL_WEIGHTS_PATH):
        self.log_file = log_file
        self.model_path = model_path
        self.printl = self._print_log
        self.model = self._load_model()

    def _print_log(self, *args):
        msg = " ".join(map(str, args))
        print(msg)
        with open(self.log_file, 'a') as f:
            f.write(msg + "\n")

    def _load_model(self):
        self.printl(f"Loading TensorFlow model on {DEVICE}...")
        model = create_model()
        # Build model with dummy input to Initialize weights structure
        dummy_input = tf.zeros((1, CONTEXT_LEN, 3), dtype=tf.int32)
        model(dummy_input)

        if os.path.exists(self.model_path):
            self.printl(f"Loading weights from {self.model_path}")
            model.load_weights(self.model_path)
        else:
            self.printl(f"Warning: Weights not found at {MODEL_WEIGHTS_PATH}. Using random weights.")

        return model

    def compress(self, input_file):
        start_time = time.time()
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        self.printl(f"Processing {base_name}...")

        # 1. Prepare Data (Generate or Load FlatBuffer)
        if input_file.endswith('.fb'):
            self.printl(f"Using existing FlatBuffer: {input_file}")
            fb_file = input_file
            # Dummy points for validation if starting from FB
            normalize_pt = []
        else:
            self.printl(f"Preparing data from raw file: {input_file}")
            # Using existing dataPrepare logic
            fb_file, dq_pt, normalize_pt = dataPrepare(
                input_file,
                saveMatDir='./Data/testPly_TF',
                offset='min',
                qs=2 / (2 ** 12 - 1),
                rotation=False,
                normalize=True
            )

        # 2. Load FlatBuffer
        with open(fb_file, 'rb') as f:
            buf = f.read()
            dataset = Dataset.Dataset.GetRootAsDataset(buf, 0)

        nodes_len = dataset.NodesLength()
        self.printl(f"Loaded encoded FB. Total nodes: {nodes_len}")
        t_load = time.time()
        self.printl(f"  [Profile] FB Load Time:      {t_load - start_time:.4f}s")

        # 3. Extract Features for Batch Inference
        features = np.zeros((nodes_len, CONTEXT_LEN, 3), dtype=np.int32)
        targets = np.zeros(nodes_len, dtype=np.int32)

        for i in range(nodes_len):
            node = dataset.Nodes(i)

            # Reconstruct Context
            # Optimized: Use FlatBuffers Numpy accessor
            neighbors = node.NeighborOccupanciesAsNumpy()
            neighbors_len = len(neighbors)

            # Optimized: Write directly to pre-allocated features array
            # Dynamic slicing based on CONTEXT_LEN
            mid = CONTEXT_LEN // 2
            needed = CONTEXT_LEN - 1

            if neighbors_len >= needed:
                features[i, 0:mid, 0] = neighbors[0:mid]
                features[i, mid, 0] = node.ParentOccupancy()
                features[i, mid+1:, 0] = neighbors[mid:]
            else:
                 features[i, mid, 0] = node.ParentOccupancy()

            features[i, :, 1] = max(0, node.Level() - 1)
            features[i, :, 2] = node.Octant()

            # Target is the occupancy code 1-255
            targets[i] = node.Occupancy()

        t_feat = time.time()
        self.printl(f"  [Profile] Feature Exc. Time: {t_feat - t_load:.4f}s")

        # 4. Model Inference (Batch Processing)
        batch_size = 512
        probs_all = np.zeros((nodes_len, 256), dtype=np.float32)

        # TF Inference
        ds = tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

        cursor = 0
        for batch_feat in tqdm.tqdm(ds, desc="Inferring"):
            logits = self.model(batch_feat, training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            probs_all[cursor : cursor + len(probs)] = probs
            cursor += len(probs)

        t_infer = time.time()
        self.printl(f"  [Profile] Inference Time:    {t_infer - t_feat:.4f}s")

        # 5. Arithmetic Encoding
        output_bin = f"{EXP_NAME}/data/{base_name}.bin"
        os.makedirs(os.path.dirname(output_bin), exist_ok=True)

        ac = numpyAc.arithmeticCoding()

        # Prepare PDF for numpyAc: [N, 256].
        # Ensure index 0 (unused class) is zeroed out if model predicts it,
        # though well-trained model should have near-zero prob.
        # To match encoder.py behavior where class 0 is explicitly 0 prob:
        probs_all[:, 0] = 0.0

        full_probs = probs_all
        full_probs += 1e-9 # Stability
        full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)

        targets_int = targets.astype(np.int16)

        byte_stream, real_bits = ac.encode(full_probs, targets_int, binfile=output_bin)
        file_size = len(byte_stream)

        t_enc = time.time()
        self.printl(f"  [Profile] Arith. Enc. Time:  {t_enc - t_infer:.4f}s")

        # 6. Stats & Verification (De-quantization error check)
        if len(normalize_pt) > 0:
            bpp = (file_size * 8) / len(normalize_pt)
        else:
            bpp = 0.0 # Unknown point count

        duration = time.time() - start_time

        self.printl(f"Compressed {base_name}: {bpp:.4f} bpp, Size: {file_size} bytes, Time: {duration:.2f}s")
        self.printl(f"Output saved to {output_bin}")

        return bpp, file_size

if __name__ == "__main__":
    os.makedirs(EXP_NAME, exist_ok=True)
    encoder = EncoderTF(f"{EXP_NAME}/encoder_log.txt")

    # List of files to process (Edit as needed)
    # Using the existing FB file found in workspace
    list_orifile = ['Data/Lidar/train/Kitti_00002510.fb']

    for ori_file in list_orifile:
        if os.path.exists(ori_file):
            encoder.compress(ori_file)
        else:
            print(f"File not found: {ori_file}")
