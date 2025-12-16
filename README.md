# ABOC: Ancestor-Based Octree Compression

**ABOC** is a modernized, TensorFlow-based implementation of **OctAttention**, designed for high-performance point cloud compression with large-scale context windows.

This project builds upon the original [OctAttention](http://arxiv.org/abs/2202.06028) (AAAI 2022) but introduces significant architectural improvements for scalability, cloud integration, and ablation testing.

## Key Differences from OctAttention

1.  **TensorFlow 2.x / Keras Implementation**: The core model has been ported from PyTorch to TensorFlow/Keras, enabling TPU compatibility and seamless integration with Google Colab.
2.  **FlatBuffers Data Format**: Replaced legacy `.mat` files with **Google FlatBuffers**. This allows for:
    *   Zero-copy deserialization.
    *   Efficient random access to large datasets (critical for handling 100k+ node trees).
    *   Significant reductions in data loading time during training and inference.
3.  **Dynamic Large-Scale Contexts**: Supports extremely large context windows (e.g., 1024+ neighbors) with efficient vectorized extraction, scalable configuration via `config.py`.
4.  **Centralized Configuration**: All hyperparameters (Context Size, Depth, Model Dims, Cloud Paths) are managed in `config.py`.
5.  **Cloud-Native Benchmarking**: dedicated notebooks (`colab_benchmark.ipynb`, `colab_training.ipynb`) for training and benchmarking directly on Google Cloud Storage (GCS) data.

## Requirements

*   Python 3.8+
*   TensorFlow 2.x
*   NumPy < 2 (Required for compatibility with certain libraries)
*   FlatBuffers (`flatc` compiler and python runtime)
*   `plyfile`, `hdf5storage`, `ninja`

### Installation

1.  **Install Python Dependencies**:
    ```bash
    pip install "numpy<2" flatbuffers plyfile Ninja hdf5storage tensorflow
    ```

2.  **Compile C++ Backend**:
    The arithmetic coder uses a C++ backend.
    ```bash
    cd numpyAc/backend
    python setup.py install
    ```
    *Note: On Google Colab, this is handled automatically in the notebook.*

## Configuration

All experiment parameters are defined in `config.py`. Modify this file to change:

*   **Data parameters**: `CONTEXT_RANGE` (neighbors), `MAX_OCTREE_LEVEL`.
*   **Model Architecture**: `EMBED_DIM`, `NUM_LAYERS`, `NUM_HEADS`.
*   **Cloud settings**: `BUCKET_NAME`, `GCS_DATA_PREFIX`.

**Example `config.py`:**
```python
CONTEXT_RANGE = 512  # Total Context = 1025
MAX_OCTREE_LEVEL = 12
BUCKET_NAME = "mtn_fb_file_bucket"
```

## Workflow

### 1. Data Preparation
Convert `.ply` point clouds into the efficient `.fb` (FlatBuffer) format. This step generates the octree structure and neighbor contexts.

```bash
# Prepare data using settings from config.py
python benchmark_data_prep.py
```

### 2. Training
Training is best performed using the provided Colab notebook to leverage GPU/TPU resources.
*   Open `colab_training.ipynb` in Google Colab.
*   Ensure `config.py` is uploaded.
*   Run the training loop (fetches data from GCS, trains TF model, saves checkpoints to GCS).

### 3. Benchmarking (Encoding/Decoding)
Run the full compression pipeline (Encode -> Bitstream -> Decode) to measure Bitrate (BPP) and Speed.

**Cloud Benchmark**:
Use `colab_benchmark.ipynb` to benchmark against a dataset in GCS.

**Local Benchmark**:
```bash
# Encode a single file
python encoder_tf.py

# Decode (Reconstruct PLY)
python decoder_tf_parallel.py
```
*Note: Ensure `encoder_tf.py` and `decoder_tf_parallel.py` are pointing to your target file or edit the `__main__` block.*

## Citation

This project is based on the research presented in **OctAttention**. If you use this work, please cite the original paper:

```bibtex
@article{OctAttention,
    title={OctAttention: Octree-Based Large-Scale Contexts Model for Point Cloud Compression},
    volume={36},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/19942},
    DOI={10.1609/aaai.v36i1.19942},
    number={1},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Fu, Chunyang and Li, Ge and Song, Rui and Gao, Wei and Liu, Shan},
    year={2022},
    month={Jun.},
    pages={625-633}
}
```
