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
            batch_features = []
            batch_meta = [] # To track (parent_idx, octant) for ordered decoding

            # Step A: Prepare batch inputs for all children of parents in the current level.
            for parent_idx in current_level_indices:
                parent_occ = AllOcts[parent_idx]

                # Check which children exist based on the parent's occupancy code.
                # `dec2bin` returns bits [7, 6, ..., 0]. We need [0, 1, ..., 7].
                children_existence_msb_first = dec2bin(parent_occ, 8)
                children_existence = children_existence_msb_first[::-1] # Reverse to get LSB first (octant 0 to 7)

                # Get Context for Parent P (neighbors in AllOcts)
                # The context window is 17 nodes centered around the parent.
                # `p_idx` is the 0-based index of the parent in `AllOcts`.

                # Calculate the slice range for neighbors in `AllOcts`.
                ctx_start = max(0, parent_idx - 8)
                ctx_end = min(len(AllOcts), parent_idx + 9) # +9 for exclusive end

                # Create a padded context array of size 17.
                full_ctx = np.zeros(17, dtype=int)

                # Map the available neighbors to the `full_ctx` array.
                # The parent itself (at `parent_idx`) maps to the center of `full_ctx` (index 8).
                for k in range(ctx_start, ctx_end):
                    target_idx = 8 + (k - parent_idx)
                    full_ctx[target_idx] = AllOcts[k]

                for octant in range(8):
                    if children_existence[octant] == 1:
                        # This child exists. We need to predict its occupancy code.
                        # Construct the input feature vector for the model.
                        # Input: [Occupancy_Context, Level_Context, Octant_Context]

                        inp_feat = np.zeros((17, 3), dtype=int)
                        inp_feat[:, 0] = full_ctx # Occupancy context
                        inp_feat[:, 1] = current_level + 1 # Level of the *child* being predicted
                        inp_feat[8, 2] = octant # Octant of the child being predicted (at the center)

                        batch_features.append(inp_feat)
                        batch_meta.append((parent_idx, octant)) # Store metadata for sequential decoding

            if not batch_features:
                break # No children found for the current level, stop decoding.

            # Step B: Parallel Prediction of probabilities for all children in the batch.
            # Convert the list of feature arrays into a single PyTorch tensor.
            batch_tensor = torch.LongTensor(np.array(batch_features)).to(device).permute(1, 0, 2) # [17, Batch, 3]

            # Perform model inference.
            output = model(batch_tensor, None, []) # [17, Batch, ntokens]

            # Extract logits for the central node (the child being predicted).
            logits = output[8, :, :] # [Batch, ntokens]
            probs_batch = torch.softmax(logits, dim=1).cpu().detach().numpy() # [Batch, ntokens]

            # Step C: Sequential Decoding using Arithmetic Decoder.
            # The arithmetic decoder requires probabilities one by one.
            for b_idx in range(len(batch_features)):
                # Decode the occupancy code for the current child.
                # `numpyAc.decode` expects a 2D array for probabilities.
                child_occ_code = ac.decode(np.expand_dims(probs_batch[b_idx], 0)) + 1 # +1 if codes are 1-255

                # Add the decoded occupancy code to the global list.
                AllOcts.append(child_occ_code)

                # Add the index of this newly decoded node to the next level's list of parents.
                next_level_indices.append(len(AllOcts) - 1)

            current_level_indices = next_level_indices
            current_level += 1

    print(f"Decoding finished in {time.time() - start:.2f} seconds.")

    # 5. Reconstruct Point Cloud from AllOcts
    # The `DeOctree` function takes the linearized occupancy codes and reconstructs the point cloud.
    # The `AllOcts` list now contains the full sequence of occupancy codes.
    ptrec = DeOctree(AllOcts)

    return ptrec

def main():
    # Initialize the Transformer model with the defined global parameters.
    model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)

    # Load pre-trained model weights.
    # This path needs to be correct for your saved model.
    # Example: model.load_state_dict(torch.load("path/to/your/model_weights.pth"))
    # For now, we'll assume a dummy model or a model loaded from `encoder.py` if it's the same.
    # If `encoder.model` is the trained model, we can use it.
    # Assuming `encoder.model` is the trained model instance.
    # If it's just the class, we need to load state_dict.

    # For this refactoring, we'll use the `encoder_model` if it's a loaded instance,
    # or load a state_dict if `model` is a fresh instance of `TransformerModel`.
    # Let's assume `encoder_model` is the trained model instance from `encoder.py`.
    # If `encoder.py` defines `model` as a class, then we need to load state_dict.
    # For now, we'll use the `model` instance created above and assume it's ready.

    # Example of loading state_dict if needed:
    # model_path = "path/to/your/trained_model.pth"
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     model.eval()
    # else:
    #     print(f"Warning: Model weights not found at {model_path}. Using uninitialized model.")

    # Set model to evaluation mode
    model.eval()

    # Loop through original files to simulate decoding process
    # `list_orifile` and `expName` would typically be defined or passed.
    # For this example, we'll use placeholders.

    # Placeholder for `expName`
    expName = "decoded_output"
    os.makedirs(expName + "/test", exist_ok=True)

    # Assuming `list_orifile` is available from `encoder.py` or defined elsewhere.
    # If `list_orifile` is empty or not defined, this loop won't run.
    if not list_orifile:
        print("No original files found in list_orifile. Decoding a dummy file.")
        # Example for a single dummy file
        dummy_bin_file = expName + "/data/dummy.bin"
        # Create a dummy bin file for testing if it doesn't exist
        os.makedirs(expName + "/data", exist_ok=True)
        with open(dummy_bin_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f') # Some dummy bytes

        # Call decodeOct with the dummy file
        reconstructed_points = decodeOct(model, dummy_bin_file)
        print(f"Decoded dummy file. Reconstructed points (first 5): {reconstructed_points[:5]}")

        # Further processing (dequantization, saving, error calculation) would go here.
        # For a dummy file, this might not be meaningful.

    else:
        for oriFile in list_orifile:
            ptName = os.path.basename(oriFile)[:-4]
            matName = 'Data/testPly/' + ptName + '.mat'
            binfile = expName + '/data/' + ptName + '.bin' # This binfile should exist from encoding

            # Call the new decodeOct function
            reconstructed_points = decodeOct(model, binfile)

            # Load original data for comparison (if available)
            # This part assumes `matloader` and `cell` structure from original code.
            # If `matName` doesn't exist, this will fail.
            try:
                cell, mat = matloader(matName)
                # Original point cloud for comparison
                p = np.transpose(mat[cell[1, 0]]['Location'])
                offset = np.transpose(mat[cell[2, 0]]['offset'])
                qs = mat[cell[2, 0]]['qs'][0]

                # Dequantization
                DQpt = (reconstructed_points * qs + offset)

                # Save reconstructed point cloud
                output_ply_path = expName + "/test/" + ptName + "_rec.ply"
                pt.write_ply_data(output_ply_path, DQpt)
                print(f"Saved reconstructed PLY to {output_ply_path}")

                # Calculate PC error
                # This assumes `pt.pcerror` is available and configured.
                # pt.pcerror(p, DQpt, None, '-r 1', None).wait()

            except FileNotFoundError:
                print(f"Warning: Original MAT file not found for {ptName}. Skipping dequantization and error calculation.")
                # If no original mat file, just save the raw reconstructed points if `ptrec` is meaningful
                output_ply_path = expName + "/test/" + ptName + "_raw_rec.ply"
                # Assuming `reconstructed_points` is already in a suitable format for `write_ply_data`
                # If `reconstructed_points` is just a list of occupancy codes, it needs conversion to actual points.
                # The `DeOctree` function should handle this.
                # For now, if `reconstructed_points` is empty, this will not save a meaningful file.
                if reconstructed_points is not None and len(reconstructed_points) > 0:
                    pt.write_ply_data(output_ply_path, reconstructed_points)
                    print(f"Saved raw reconstructed PLY to {output_ply_path}")


if __name__ == "__main__":
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
