import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Hyperparameters from config ---
from config import CONTEXT_LEN, MAX_OCTREE_LEVEL, VOCAB_SIZE, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, DROPOUT

def create_model():
    """
    Creates the ABOC Transformer model using TensorFlow/Keras.
    Inputs: [Batch, Sequence=1025, Channels=3] (Occupancy, Level, Octant)
    Output: [Batch, VOCAB_SIZE] (Logits for center node occupancy)
    """
    inputs = keras.Input(shape=(CONTEXT_LEN, 3), dtype=tf.int32)

    # Split channels
    occ_input = inputs[:, :, 0]
    lvl_input = inputs[:, :, 1]
    oct_input = inputs[:, :, 2]

    # Embeddings (Matching octAttention.py)
    # self.encoder = nn.Embedding(ntoken, 130)
    occ_emb = layers.Embedding(VOCAB_SIZE, 130)(occ_input)

    # self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL + 1, 6)
    lvl_input = layers.Lambda(lambda x: tf.clip_by_value(x, 0, MAX_OCTREE_LEVEL))(lvl_input)
    lvl_emb = layers.Embedding(MAX_OCTREE_LEVEL + 1, 6)(lvl_input)

    # self.encoder2 = nn.Embedding(9, 4)
    oct_emb = layers.Embedding(9, 4)(oct_input)

    # Concatenate: 130 + 6 + 4 = 140
    x = layers.Concatenate(axis=-1)([occ_emb, lvl_emb, oct_emb])

    # Custom Layer to handle SparseTensor from TPU Embeddings (or general robustness)
    class ToDense(layers.Layer):
        def __init__(self, **kwargs):
            super(ToDense, self).__init__(**kwargs)

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, inputs):
            # Check for SparseTensor
            if isinstance(inputs, tf.SparseTensor):
                return tf.sparse.to_dense(inputs)
            # Check for KerasTensor's sparse property
            if hasattr(inputs, 'sparse') and inputs.sparse:
                return tf.sparse.to_dense(inputs)
            return inputs

    x = ToDense()(x)

    # Scale by sqrt(embedding_dim) - PyTorch does this in forward
    x = x * tf.math.sqrt(tf.cast(140.0, tf.float32))

    # Positional Encoding
    class PositionalEncoding(layers.Layer):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            self.d_model = d_model
            # Compute PE once
            position = np.arange(max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            pe = np.zeros((max_len, d_model))
            pe[:, 0::2] = np.sin(position * div_term)
            pe[:, 1::2] = np.cos(position * div_term)
            self.pe = tf.constant(pe, dtype=tf.float32)

        def compute_output_shape(self, input_shape):
            return input_shape

        def call(self, x):
            # x shape: [Batch, Seq, Dim]
            seq_len = tf.shape(x)[1]
            # Use dynamic sizing to avoid errors if batch has smaller seq
            pe_slice = self.pe[:seq_len, :]
            return x + pe_slice

    x = PositionalEncoding(EMBED_DIM)(x)
    if DROPOUT > 0:
        x = layers.Dropout(DROPOUT)(x)

    # Transformer Encoder
    for _ in range(NUM_LAYERS):
        # MultiHead Attention
        # key_dim = embed_dim // num_heads
        att_output = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM//NUM_HEADS)(x, x)
        if DROPOUT > 0:
            att_output = layers.Dropout(DROPOUT)(att_output)
        x1 = layers.LayerNormalization(epsilon=1e-6)(x + att_output)

        # Feed Forward
        ffn_output = layers.Dense(FF_DIM, activation='relu')(x1)
        ffn_output = layers.Dense(EMBED_DIM)(ffn_output)
        if DROPOUT > 0:
            ffn_output = layers.Dropout(DROPOUT)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_output)

    # Decoder Head
    # self.decoder0 = nn.Linear(ninp, ninp)
    x = layers.Dense(EMBED_DIM, activation='relu')(x)

    # self.decoder1 = nn.Linear(ninp, ntoken)
    logits = layers.Dense(VOCAB_SIZE)(x)

    # Extract center token (Index 512 in sequence of 1025)
    # Since CONTEXT_LEN=1025, mid index is 512
    center_idx = CONTEXT_LEN // 2
    center_logits = logits[:, center_idx, :]

    return keras.Model(inputs=inputs, outputs=center_logits)
