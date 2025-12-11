"""
Author: fuchy@stu.pku.edu.cn
Description: this file is the encoder helper.
FilePath: /compression/encoderTool.py
All rights reserved.
"""

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


def compress(oct_data_seq, outputfile, model, actualcode=True, print=print, showRelut=False):
    model.eval()

    # Needs to reconstruct Octree from oct_data_seq to get ParallelContext?
    # oct_data_seq comes from 'dataset.py' logic usually, but here main() passes specific legacy format.
    # main() extracts: oct_data_seq = np.transpose(mat[cell[0, 0]]).astype(int)[:, -FeatDim:, 0:6]
    # This is the "K-Parent" sequence.
    # We need the full Octree structure to generate Parallel Context.

    # We should modify 'main' or 'compress' to accept the full Octree or reconstruct it.
    # 'oct_data_seq' contains [Oct, Level, Octant, Pos] ?
    # Shape of oct_data_seq in main: [N, K, 6]?
    # Actually main extracts "[:, -FeatDim:, 0:6]".
    # We might need to reload the .mat more carefully or just expect the octree structure.
    # BUT: The user moved to FlatBuffers.
    # 'encoder.py' calls 'dataPrepare' which now saves .fb files.
    # 'main' in 'encoder.py' loads the file.
    # 'main' needs to be updated to load .fb if available.

    # Assuming we have the Octree nodes in order (Linear).
    # We need to reconstruct the "Context" array.
    # If we have the list of Occupancies + Levels, we can reconstruct the context (Parent + Neighbors).
    # Since the input is linearized Octree (Breadth First or Depth First? Morton?), we can deduce neighbors?
    # Morton order is standard.
    # "Parent + Neighbors" implies we need the parent's index in the previous level.
    # It's complex to reconstruct from just the sequence without the tree pointers.
    # The 'Octree.py' `GenOctree` returns the `Octree` object which has structure.

    # Strategy:
    # 1. Update 'main' to pass the loaded FlatBuffer data (which computes Context in data loader!)
    # 2. Or if running from .mat/scratch, re-run GenOctree?
    # encoder.py calls dataPrepare. dataPrepare returns (file, DQpt, ref).
    # dataPrepare computes GenOctree internally.
    # We can't easily get the Octree object back from dataPrepare unless we change return.
    # However, 'dataset.py' loads the file and returns the context!
    # So 'encoderTool.main' should use 'dataset.py' to load the data.

    # Let's assume 'oct_data_seq' passed here is actually the "Nodes" list with Context already computed?
    # No, 'encoder.py' passes result of 'matloader'.
    # I should modify 'encoder.py' to use the new 'dataset' FlatBuffer loader.
    # AND modify 'compress' to accept the batch of data [N, 17, 3/4] directly.

    # Let's write 'compress' assuming it receives:
    # `nodes_features`: [N, 17, 3] (Context, Level, Octant) - Ready for model.
    # `targets`: [N] (Occupancy) - For coding.

    # We assume `oct_data_seq` is actually a tuple (features, targets) or similar.
    # Wait, 'compress' signature is fixed called by encoder.py.
    # I will update encoder.py too.

    nodes_features = oct_data_seq['features'] # [N, 17, 3]
    targets = oct_data_seq['targets'] # [N]

    # Batch processing
    batch_size = 512
    num_nodes = nodes_features.shape[0]

    proBit = []

    with torch.no_grad():
        for i in range(0, num_nodes, batch_size):
            end = min(i + batch_size, num_nodes)
            batch_input = nodes_features[i:end] # [B, 17, 3]

            # Model Forward
            # Input to model: [17, B, 3]
            model_input = torch.from_numpy(batch_input).long().to(device).permute(1, 0, 2)

            output = model(model_input, None, []) # [17, B, 255]

            # Prediction is at center index 8
            output = output[8, :, :] # [B, 255]

            p = torch.softmax(output, 1) # [B, 255]
            proBit.append(p.cpu().numpy())

    proBit = np.vstack(proBit) # [N, 255]

    # Arithmetic Coding
    bit = 0
    oct_len = num_nodes

    # Estimate bits
    # for i in range(oct_len):
    #     octvalue = targets[i]
    #     bit += -np.log2(proBit[i, octvalue-1] + 1e-7)

    # Vectorized estimate
    row_indices = np.arange(oct_len)
    col_indices = targets - 1
    # Clip indices to be safe
    col_indices = np.clip(col_indices, 0, 254)
    probs = proBit[row_indices, col_indices]
    bits = -np.log2(probs + 1e-7)
    binsz = np.sum(bits)

    elapsed = 0 # Placeholder

    if actualcode:
        codec = numpyAc.arithmeticCoding()
        if not os.path.exists(os.path.dirname(outputfile)):
            os.makedirs(os.path.dirname(outputfile))
        # Encoder/Decoder expects occupancy code 1..255?
        # Check numpyAc usage.
        # "oct_seq.astype(np.int16).squeeze(-1) - 1" -> 0..254
        _, _ = codec.encode(proBit, targets.astype(np.int16) - 1, outputfile)

    # Return summary stats (simplified)
    return binsz, oct_len, elapsed, np.array([]), np.array([])
        # %%


def main(data, model, actualcode=True, showRelut=True, printl=print):
    # data is expected to be a dict: {'features': np.array, 'targets': np.array, 'ptName': str, 'rawVal': any}

    ptName = data.get('ptName', 'unknown')
    outputfile = expName + "/data/" + ptName + ".bin"

    binsz, oct_len, elapsed, binszList, octNumList = compress(data, outputfile, model, actualcode, printl, showRelut)

    if showRelut:
        printl("ptName: ", ptName)
        printl("binsize(b):", binsz)
        printl("oct len", oct_len)
        printl("bit per oct:", binsz / oct_len if oct_len > 0 else 0)

    return binsz / oct_len if oct_len > 0 else 0
