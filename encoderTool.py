#%%
import numpy as np
import torch
import time
import os
from networkTool import device, bptt, expName, levelNumK, MAX_OCTREE_LEVEL
from dataset import default_loader as matloader
import numpyAc
import tqdm

bpttRepeatTime = 1


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


#%%

'''
description: Rearrange data for batch processing
'''


def dataPreProcess(oct_seq, bptt, batch_size, oct_len):
    oct_seq[:, :, 0] = oct_seq[:, :, 0] - 1
    oct_seq = torch.Tensor(oct_seq).long()  # [1,255]->[0,254]. shape [n,K]
    FeatDim = oct_seq.shape[-1]
    padingdata = torch.vstack((torch.zeros((bptt, levelNumK, FeatDim)), oct_seq))  # to be the context of oct[0]
    padingsize = batch_size - padingdata.shape[0] % batch_size
    padingdata = torch.vstack((padingdata, torch.zeros(padingsize, levelNumK, FeatDim))).reshape(
        (batch_size, -1, levelNumK, FeatDim)).permute(1, 0, 2, 3)  #[bptt,batch_size,K]
    dataID = torch.hstack(
        (torch.ones(bptt) * -1, torch.Tensor(list(range(oct_len))), torch.ones((padingsize)) * -1)).reshape(
        (batch_size, -1)).long().permute(1, 0)
    padingdata = torch.vstack((padingdata, padingdata[0:bptt, list(range(1, batch_size, 1)) + [0]])).long()
    dataID = torch.vstack((dataID, dataID[0:bptt, list(range(1, batch_size, 1)) + [0]])).long()
    return dataID, padingdata


def encodeNode(pro, octvalue):
    assert 255 >= octvalue >= 1
    pre = np.argmax(pro) + 1
    return -np.log2(pro[octvalue - 1] + 1e-07), int(octvalue == pre)


'''
description: compress function
param {N[n treepoints]*K[k ancestors]*C[oct code,level,octant,position(xyz)] array; Octree data sequence} oct_data_seq
param {str;bin file name} outputfile
param {model} model
param {bool;Determines whether to perform entropy coding} actualcode
return {float;estimated/true bin size (in bit)} binsz
return {int;oct length of all octree} oct_len
return {float;total foward time of the model} elapsed
return {float list;estimated bin size (in bit) of depth 8~maxlevel data} binszList
return {int list;oct length of 8~maxlevel octree} octNumList
'''



# Ensure Octree and Dataset modules are available
try:
    import Octree
    import OctreeData.Dataset as Dataset
    import OctreeData.OctreeNode as OctreeNode
except ImportError:
    pass # Might be running in an environment without these explicitly set up yet

from octAttention import TransformerModel
from Octree import Morton, DeOctree

# Globals for model configuration
ntokens = 255
ninp = 128 + 4 + 6
nhid = 300
nlayers = 3
nhead = 4
dropout = 0


def compress(data, model, outputfile, actualcode=True):
    """
    Compresses the octree data using Arithmetic Coding and the trained Transformer model.

    Args:
        data (dict): Dictionary containing prepared data:
            - features: [N, 17, 3] input features (Occupancy, Level, Octant)
            - targets: [N] target occupancies to encode
            - ptName: Name of the point cloud
            - visualize_data: tuple (points, code) for verification
        model (nn.Module): The trained Transformer model.
        outputfile (str): Path to save the compressed binary file.
        actualcode (bool): Whether to perform actual arithmetic coding.

    Returns:
        float: Bits per point (bpp) achieved.
        float: Encoding time in seconds.
        int: Size of the compressed file in bytes.
    """
    start = time.time()

    # 1. Initialize Arithmetic Encoder
    ac = numpyAc.arithmeticCoding()

    # 2. Extract Data
    features = data['features'] # [N, 17, 3]
    targets = data['targets']   # [N]
    num_nodes = len(targets)

    print(f"Compressing {num_nodes} nodes to {outputfile}...")

    # 3. Model Inference to get Probabilities
    # We predict probabilities for all nodes.
    # To avoid OOM, we can batch the inference.
    batch_size = 512
    proBit = np.zeros((num_nodes, 255), dtype=np.float32) # Store probs for classes 1-255

    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, num_nodes, batch_size), desc="Inferring probabilities"):
            end = min(i + batch_size, num_nodes)
            batch_img = features[i:end] # [B, 17, 3]

            # Convert to Tensor [17, B, 3]
            input_tensor = torch.from_numpy(batch_img).permute(1, 0, 2).float().to(device)

            # Predict
            output = model(input_tensor, None, []) # [17, B, 255]

            # We want the output corresponding to the center node (index 8)
            output = output[8, :, :] # [B, 255]

            # Softmax to get probabilities
            prob = torch.softmax(output, dim=1).cpu().numpy() # [B, 255]

            proBit[i:end] = prob

    # 4. Arithmetic Encoding
    # Encode targets sequentially using pre-computed probabilities

    # Check if we should use actual coding
    if actualcode:
        # Prepare probability table: [N, 256] -> padding index 0 with zeros?
        # numpyAc expects a specific format.
        # Based on `numpyAc.py`: `encode(pdf, sym)`
        # pdf should be [N, 256] (if symDim=255?).
        # targets are in range [1, 255].
        # So we need a distribution of size 256.
        # Index 0 is probability of symbol 0 (which is unused/dummy).

        full_probs = np.zeros((num_nodes, 256), dtype=np.float32)
        full_probs[:, 1:] = proBit # Map 1-255 predictions to indices 1-255

        # Ensure sum is 1.0 (might need normalization if float errors)
        # Adding epsilon to avoid zero prob for any symbol potentially?
        # numpyAc handles normalization internally if we pass unnormalized.
        # But let's keep it clean.
        full_probs += 1e-9
        full_probs = full_probs / full_probs.sum(axis=1, keepdims=True)

        targets_int = targets.astype(np.int16)

        # Encode!
        byte_stream, real_bits = ac.encode(full_probs, targets_int, binfile=outputfile)

        file_size = len(byte_stream)

    else:
        # Simulation only
        total_bits = 0
        for i in range(num_nodes):
            target_code = int(targets[i])
            if target_code > 0:
                p = proBit[i, target_code - 1] # proBit is 0-indexed for classes 1-255
                total_bits += -np.log2(p + 1e-9)

        file_size = int(total_bits / 8)
        # Write dummy file if requested? Or just skip.
        if outputfile:
            with open(outputfile, 'wb') as f:
                f.write(b'SIMULATION')

    end = time.time()
    encodetime = end - start

    # Calculate Bits Per Point
    # Verification data
    points, codes = data['visualize_data']

    bpp = (file_size * 8) / len(points)
    print(f"Compressed {data.get('ptName', 'Point Cloud')}: {bpp:.4f} bpp, Size: {file_size} bytes, Time: {encodetime:.2f}s")

    return bpp, encodetime, file_size


def main(data_dict):
    """
    Main function for compression.

    Args:
        data_dict: Dictionary containing features and targets.
    """
    # Load Model
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)

    # Load Weights
    if os.path.exists(checkpoint_file):
        print(f"Loading model from {checkpoint_file}")
        state = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state['encoder'])
    else:
        print("No checkpoint found! Using random weights (for testing only).")

    # Compress
    compress(data_dict, model)

if __name__ == "__main__":
    # Test stub
    pass
