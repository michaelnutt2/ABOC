
"""
ABOC Configuration File
Centralizes hyperparameters for ablation testing regarding context size, depth,
and model architecture.
"""

# --- Data Configuration ---
# Parallel Context Range: Number of neighbors on EACH side of the parent.
# Total Context Length = (2 * CONTEXT_RANGE) + 1 (Parent)
CONTEXT_RANGE = 64

# Derived Context Length (e.g., 512*2 + 1 = 1025)
CONTEXT_LEN = (2 * CONTEXT_RANGE) + 1

# Maximum Octree Depth for encoding/decoding
MAX_OCTREE_LEVEL = 12

# Vocabulary Size (Occupancy codes 0-255)
VOCAB_SIZE = 256

# --- Model Architecture ---
# Transformer Hyperparameters
EMBED_DIM = 140 # Embedding Dimension (130 + 6 + 4)
NUM_HEADS = 4   # Number of Attention Heads
FF_DIM = 300    # Feed Forward Network Dimension
NUM_LAYERS = 3  # Number of Transformer Layers
DROPOUT = 0.0   # Dropout Rate

# --- Training / Experiment ---
EXP_NAME = './Exp/Kitti_TF'
BATCH_SIZE = 32
EPOCHS = 50

# --- Cloud Configuration ---
BUCKET_NAME = "mtn_fb_file_bucket"
# GCS_DATA_PREFIX = "data"
GCS_DATA_PREFIX = "data_subset_64"
GCS_CHECKPOINT_PREFIX = "checkpoints"