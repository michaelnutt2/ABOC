"""
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
"""
#%%
import numpy as np
import torch
from tqdm import tqdm
from Octree import DeOctree, dec2bin
import pt
from dataset import default_loader as matloader
from collections import deque
import os
import time
from networkTool import *
from encoderTool import bpttRepeatTime, generate_square_subsequent_mask
from encoder import model, list_orifile
import numpyAc

batch_size = 1

#%%
'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''


def decodeOct(binfile, oct_data_seq, model, bptt):
    model.eval()
    elapsed = time.time()

    # Initialize Arithmetic Decoder
    # oct_len is needed for decoder?
    # standard arithmetic decoder needs length, but we can manage context dynamically.
    # The provided 'numpyAc' seems to need total length.
    oct_len = len(oct_data_seq)
    dec = numpyAc.arithmeticDeCoding(None, oct_len, 255, binfile)

    # Context State
    # We decod level by level.
    # Level 0: Root (always 255/Occupied?).
    # In this dataset/code, Root is usually handled or we start from Level 1?
    # Original code:
    # KfatherNode = ...
    # oct_seq.append(root)

    # Let's assume Root is first node and is known/decoded first.
    # How do we get prob for Root?
    # Root context is zeros?
    # output = model(input, src_mask, [])
    # freqsinit = ...
    # root = decodeNode(freqsinit, dec)

    # Initial Context (Zeros)
    # [17, B, 3]. Batch=1 for root.
    init_context = torch.zeros((17, 1, 3)).long().to(device)
    output = model(init_context, None, [])
    probs = torch.softmax(output[8, 0, :], 0).cpu().detach().numpy()

    root_occ = decodeNode(probs, dec)

    # Lists to store the Linear Sequence
    # We need to maintain order to match encoder (Morton).
    # We also need to build the tree to find neighbors.
    # But actually, if we just store "Current Level Nodes" in a list,
    # and they are sorted by Morton Code (which they are if we traverse 0..7),
    # then "Neighbors" are just indices in the list!

    # current_level_nodes: List of {val: int, level: int}
    # Wait, we need "Parent" reference to form next level.
    # And we need to know WHICH child of parent we are (Octant).

    decoded_seq = [root_occ]

    # Level 1 Nodes
    # Generated from Root.
    # Root has children based on 'root_occ' bits.

    # Current Level Parents: List of (NodeIndexInSeq, Occupancy)
    # We need to track global index to map back to neighbors in the flattened sequence?
    # Actually, GenParallelContext uses "Parent's +/- 8 neighbors in the Flattened Sequence".
    # So we need the Linear History of all nodes decoded so far?
    # "AllOcts" in GenParallel.
    # Context is: `AllOcts[max(1, p_id - 8) : p_id + 9]`
    # where p_id is the index of Parent in the sequence.

    # So we just need a global list `AllOcts`.
    AllOcts = [root_occ]

    # Breadth First traversal?
    # Original code used a Queue -> BFS.
    # So `AllOcts` will be populated in BFS order.
    # This matches the Encoder order if Encoder was BFS.
    # `GenKparentSeq` logic: `for L in range(LevelNum): for node in Octree[L].node`.
    # Yes, it iterates Level by Level. So it is BFS.

    # Current Frontier: List of indices in AllOcts that are in the current level.
    current_level_indices = [0] # Root is index 0

    # Max Levels?
    # We stop when queue is empty or max level reached.

    current_level = 0

    with torch.no_grad():
        while current_level_indices:
            # We have a list of parents (current level).
            # We want to decode their children (next level).

            # 1. Prepare Batch for All Children of All Parents?
            # Each Parent has 8 potential children (Octants 0-7).
            # We need to predict occupancy for each child?
            # Wait, `root_occ` is a byte (1-255). It represents WHICH children exist.
            # The Model predicts the *Occupancy Code* of the *Parent*?
            # NO.
            # The Model predicts `node.oct` based on `node.parent`'s context?
            # In `Octree.py`: `Seq[n] = node.oct`.
            # `Context` is based on `Parent's Neighbors`.

            # CORRECT LOGIC:
            # The model predicts the *Occupancy Code* of a Node X.
            # Input features are derived from *Node X's Parent* (and Parent's neighbors).
            # So, to decode Level L+1:
            # We iterate over Parents in Level L.
            # For each Parent P:
            #   We prepare context (P's neighbors in Level L).
            #   We run model -> Predict P.oct?
            #   Wait, P.oct is ALREADY DECODED (it is P's occupancy).
            #   The "Occupancy" we store in the dataset is the *Children Configuration* of the node?
            #   `node.oct` = occupancyCode 1~255.
            #   Yes. `node.oct` tells us which children exist.
            #   So, `node.oct` is a property of `node`.
            #   BUT: We predict `node.oct` using `node.parent` context?
            #   Check `GenParallelContext`:
            #   It iterates `n` (current node). Finds `p_id` (parent).
            #   Context = Neighbors of `p_id`.
            #   So yes: To predict `child.oct`, we look at `parent`.
            #   Wait, `child.oct` determines grandchildren.
            #   If we are at Root (Level 0). We know Root.
            #   Root.oct determines presence of Level 1 nodes.
            #   Level 1 nodes have `oct` values (determining Level 2).
            #   We need to predict `Level1Node.oct`.
            #   Context for `Level1Node` is `Root` (its parent) + Neighbors.
            #   Root has no neighbors.

            # So:
            # 1. We have encoded Root.oct (using dummy context).
            # 2. Decode Root.oct. It tells us we have children C1, C2... at Level 1.
            # 3. For each child Ci:
            #       Predict Ci.oct.
            #       Input: Context(Parent(Ci)) = Context(Root).
            #       Wait, all children of Root share the same Parent Context?
            #       Differentiation comes from `Octant` feature!
            #       Query = (Parent Context + Ci.Octant).
            #       This allows predicting different oct codes for different siblings.

            # So:
            # Phase 1: Identify all existing children in Next Level.
            # Phase 2: Prepare Context for all of them.
            # Phase 3: Decode their Oct codes.

            next_level_indices = []

            # Step A: Expand current level parents to find which children exist.
            # We also need to assign them indices in `AllOcts` (temporarily placeholders?)
            # No, we decode them sequentially.

            # We can't batch *everything* in one go if arithmetic coding is serial.
            # But we can batch the *Probability Computation*.

            batch_inputs = []
            metadata = [] # (ParentIndex, ChildOctant)

            for p_idx in current_level_indices:
                p_oct = AllOcts[p_idx]
                # Convert occ byte to list of existing children octants
                # dec2bin(p_oct) -> bits. If bit i is 1, child i exists.
                # Assuming standard order (0..7).
                bits = dec2bin(p_oct) # [bit7, ... bit0] or [bit0...]?
                # Octree.py: `dec2bin(n,8)`: `[(n>>y)&1 for y in range(7, -1, -1)]` -> MSB first (7 down to 0).
                # `popleft` logic in original decoder: `childOcu.reverse()`. `for i in range(8)`.
                # Original loop: `for i in range(8)`.
                # If bits is MSB first [7, 6... 0].
                # If we iterate i=0..7. We check bits[7-i]?
                # `dec2bin` returns [bit7, bit6... bit0].
                # Original decoder: `childOcu = dec2bin(father); childOcu.reverse()` -> [bit0, bit1...].
                # `for i in range(8): if childOcu[i]: child i exists`.
                # So Child 0 corresponds to LSB.

                bits = dec2bin(p_oct)
                bits.reverse() # [bit0, bit1 ... bit7]

                # Get Context for Parent P
                # p_idx is index in AllOcts.
                start = max(0, p_idx - 8) # AllOcts is 0-indexed here?
                # GenParallel used 1-based logic?
                # `p_id` in Octree was 1-based. `Octree[0].node[0].nodeid = 0?`
                # Code says `nodeid` starts at 0?
                # `Octree.py`: `if Octree and .. nodeNum = ... nodeid`.
                # `Octree[0].node[0].parent = 1`.
                # If Root is nodeid=0. Parent=1???
                # GenKparentSeq: `Octree[0].node[0].parent = 1`.
                # `Seq[0] = root.oct`.
                # If p_id refers to `AllOcts` index?
                # `Seq[node.parent - 1]` accessed. So `parent` is 1-based index into Seq?
                # If Root is at index 0. Then Root's index is 1 (in 1-based world)?
                # So `parent - 1` maps to 0. Correct.

                # So if `p_idx` is 0-based index in `AllOcts`.
                ctx_start = max(0, p_idx - 8)
                ctx_end = min(len(AllOcts), p_idx + 9)

                # Neighbor Slice
                neighbors = AllOcts[ctx_start:ctx_end]

                # Pad if boundaries
                # Need exactly 17 items. Center is P (p_idx).
                # Padding? Zeros?
                # `GenParallel`: `if p_id > 0 ... val_slice = AllOcts[start:end+1] ... Context[n, col_start:col_end] = val_slice`.
                # `AllOcts` in GenParallel was `zeros(nodeNum+1)`.
                # So padded with zeros.

                full_ctx = np.zeros(17, dtype=int)

                # Map neighbors to full_ctx
                # Center of full_ctx is 8.
                # p_idx maps to 8.
                # k maps to 8 + (k - p_idx).

                # Relative range in AllOcts: [ctx_start, ctx_end)
                for k in range(ctx_start, ctx_end):
                    target_idx = 8 + (k - p_idx)
                    full_ctx[target_idx] = AllOcts[k]

                # Current Level of Parent?
                # Need to track levels.
                # `AllOcts` only stores values.
                # We need a parallel list `AllLevels`?
                # Or just pass simple level tracking.
                # All parents in `current_level_indices` are at `current_level`.
                p_level = nodes_count # No, we need actual level.
                # We can track `current_level` var.

                for i in range(8):
                    if bits[i] == 1:
                        # Child i exists.
                        # We need to predict its Occupancy Code.
                        # Input: Context (full_ctx), Level (p_level), Octant (i).

                        # Prepare input vector [17, 3]
                        # [Occ, Lvl, Oct]
                        # Occ = full_ctx
                        # Lvl = p_level
                        # Oct = i (Center only? Or all?)
                        # As discussed, Octant is only relevant for the query (Center).
                        # Others 0.

                        inp = np.zeros((17, 3), dtype=int)
                        inp[:, 0] = full_ctx
                        inp[:, 1] = current_level
                        # Dataset: `lvl - 1`. `lvl` was Child Level. So `lvl-1` is Parent Level.
                        # So we pass `current_level`.
                        # `current_level` starts at 0 (Root).

                        # Wait, track level variable in loop.

                        inp[:, 2] = 0
                        inp[8, 2] = i # Octant of child

                        batch_inputs.append(inp)
                        metadata.append((p_idx, i)) # Metadata to track order?

            if not batch_inputs:
                break

            # Batch Inference
            # Stack inputs [Batch, 17, 3]
            # Permute to [17, Batch, 3]
            batch_tensor = torch.LongTensor(np.array(batch_inputs)).to(device).permute(1, 0, 2)

            # Add Level Info
            # We didn't fill level in loop efficiently.
            # Fill directly in tensor?
            # All have same level `current_level`.
            # batch_tensor[:, :, 1] = current_level
            # Actually, `current_level` variable needs to be maintained.

            # Need to fix Level tracking.
            # `current_level` starts 0.

            output = model(batch_tensor, None, []) # [17, B, 255]

            # Predictions at Center
            logits = output[8, :, :] # [B, 255]
            probs = torch.softmax(logits, 1).cpu().detach().numpy()

            # Decode One by One
            for b in range(len(batch_inputs)):
                # Decode
                code = decodeNode(probs[b], dec)

                # Add to AllOcts
                AllOcts.append(code)
                next_level_indices.append(len(AllOcts) - 1)

            current_level_indices = next_level_indices
            current_level += 1

    Code = AllOcts
    return Code, time.time() - elapsed


def decodeNode(pro, dec):
    root = dec.decode(np.expand_dims(pro, 0))
    return root + 1


if __name__ == "__main__":

    for oriFile in list_orifile:
        ptName = os.path.basename(oriFile)[:-4]
        matName = 'Data/testPly/' + ptName + '.mat'
        binfile = expName + '/data/' + ptName + '.bin'
        cell, mat = matloader(matName)

        # Read Sideinfo
        oct_data_seq = np.transpose(mat[cell[0, 0]]).astype(int)[:, -1:, 0]  # for check

        p = np.transpose(mat[cell[1, 0]]['Location'])  # ori point cloud
        offset = np.transpose(mat[cell[2, 0]]['offset'])
        qs = mat[cell[2, 0]]['qs'][0]

        Code, elapsed = decodeOct(binfile, oct_data_seq, model, bptt)
        print('decode succee,time:', elapsed)
        print('oct len:', len(Code))

        # DeOctree
        ptrec = DeOctree(Code)
        # Dequantization
        DQpt = (ptrec * qs + offset)
        pt.write_ply_data(expName + "/test/rec.ply", DQpt)
        pt.pcerror(p, DQpt, None, '-r 1', None).wait()
