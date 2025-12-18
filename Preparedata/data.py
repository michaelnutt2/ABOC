
from Octree import GenOctree, GenKparentSeq, GenParallelContext
import pt as pointCloud
import numpy as np
import os
import hdf5storage
from config import CONTEXT_RANGE


def dataPrepare(fileName, saveMatDir='Data', qs=1, ptNamePrefix='', offset='min', qlevel=None, rotation=False,
                normalize=False):
    if not os.path.exists(saveMatDir):
        os.makedirs(saveMatDir)
    import time

    t_start = time.time()
    ptName = ptNamePrefix + os.path.splitext(os.path.basename(fileName))[0]
    p = pointCloud.ptread(fileName)

    refPt = p
    if normalize is True:  # normalize pc to [-1,1]^3
        p = p - np.mean(p, axis=0)
        p = p / abs(p).max()
        refPt = p

    if rotation:
        refPt = refPt[:, [0, 2, 1]]
        refPt[:, 2] = - refPt[:, 2]

    if offset == 'min':
        offset = np.min(refPt, 0)

    points = refPt - offset

    if qlevel is not None:
        qs = (points.max() - points.min()) / (2 ** qlevel - 1)

    pt = np.round(points / qs)
    pt, idx = np.unique(pt, axis=0, return_index=True)
    pt = pt.astype(int)

    t_load = time.time()
    # print(f"  [Profile] Load & Preprocess: {t_load - t_start:.4f}s")

    code, Octree, QLevel = GenOctree(pt)

    t_oct = time.time()
    # print(f"  [Profile] GenOctree:         {t_oct - t_load:.4f}s")

    # DataSturct = GenKparentSeq(Octree, 4) # OLD

    # Generate Parallel Context (Parent + Neighbors)
    # Using context_range from config
    Context, AllOcts = GenParallelContext(Octree, context_range=CONTEXT_RANGE)

    t_ctx = time.time()
    # print(f"  [Profile] GenParallelContext:{t_ctx - t_oct:.4f}s")

    ptcloud = {'Location': refPt}
    # Info = {'qs': qs, 'offset': offset, 'Lmax': QLevel, 'name': ptName,
    #        'levelSID': np.array([Octreelevel.node[-1].nodeid for Octreelevel in Octree])}

    # Save as FlatBuffer
    try:
        import flatbuffers
        # Attempt to import generated classes.
        # Note: These will only exist after `flatc --python schema.fbs` is run.
        import OctreeData.Dataset as Dataset
        import OctreeData.OctreeNode as OctreeNode

        builder = flatbuffers.Builder(1024 * 1024)

        # Create Nodes
        # We need to iterate backwards to create objects before adding them to vector?
        # Or create all objects then create vector?
        # FlatBuffers requires creating nested objects (like vectors inside tables) before the table.
        # But we also need to create the 'nodes' vector for the Dataset table.

        # We have N nodes.
        # For each node, we have a 'neighbor_occupancies' vector.

        node_offsets = []

        # Helper to get node attributes. We need to iterate the Octree again or store mapping.
        # The 'Context' array corresponds to the sequence of nodes.
        # We can iterate Octree to get pos/level/octant and sync with Context row index.

        # Let's collect all node data first to iterate backwards easily
        # or just recurse/iterate backwards.
        # Since Octree is structured, we can linearize it into a list first.
        flat_nodes = []
        n = 0
        for L in range(len(Octree)):
            for node in Octree[L].node:
                flat_nodes.append({
                    'node': node,
                    'context': Context[n]
                })
                n += 1

        # Now iterate backwards
        for i in range(len(flat_nodes) - 1, -1, -1):
            item = flat_nodes[i]
            node = item['node']
            ctx = item['context']

            # Create Neighbor Vector (int array)
            # context array includes parent at center.
            # Schema has: parent_occupancy and neighbor_occupancies.
            # Let's split them.
            # ctx length is 2*CONTEXT_RANGE + 1. Center is index CONTEXT_RANGE.
            center_idx = CONTEXT_RANGE
            parent_occ = ctx[center_idx]
            neighbors = np.delete(ctx, center_idx) # Remove parent from neighbors list

            OctreeNode.OctreeNodeStartNeighborOccupanciesVector(builder, len(neighbors))
            for val in reversed(neighbors): # Prepend
                builder.PrependInt32(val)
            neighbors_vec = builder.EndVector()

            # Create Node Table
            OctreeNode.OctreeNodeStart(builder)
            OctreeNode.OctreeNodeAddPosX(builder, int(node.pos[0]))
            OctreeNode.OctreeNodeAddPosY(builder, int(node.pos[1]))
            OctreeNode.OctreeNodeAddPosZ(builder, int(node.pos[2]))
            OctreeNode.OctreeNodeAddLevel(builder, Octree[L].level if hasattr(Octree[L], 'level') else L+1) # Check level logic
            OctreeNode.OctreeNodeAddOctant(builder, node.octant)
            OctreeNode.OctreeNodeAddOccupancy(builder, node.oct)
            OctreeNode.OctreeNodeAddParentId(builder, node.parent)
            OctreeNode.OctreeNodeAddParentOccupancy(builder, parent_occ)
            OctreeNode.OctreeNodeAddNeighborOccupancies(builder, neighbors_vec)

            off = OctreeNode.OctreeNodeEnd(builder)
            node_offsets.append(off)

        # Create Nodes Vector (Prepend offsets)
        Dataset.DatasetStartNodesVector(builder, len(node_offsets))
        for off in node_offsets: # node_offsets was created backwards (last node first), but Prepend expects last element first?
            # Wait, Prepend expects the text/items in reverse order of appearance in the vector.
            # If we want Node 0, Node 1...
            # We should Prepend Node N, then Node N-1...
            # node_offsets[0] corresponds to Node N (last node).
            # So we iterate node_offsets forward.
            builder.PrependUOffsetTRelative(off)
        nodes_vec = builder.EndVector()

        # Create Name String
        name_str = builder.CreateString(ptName)

        # Create Dataset
        Dataset.DatasetStart(builder)
        Dataset.DatasetAddNodes(builder, nodes_vec)
        Dataset.DatasetAddQs(builder, float(qs))
        Dataset.DatasetAddOffsetX(builder, float(offset[0]))
        Dataset.DatasetAddOffsetY(builder, float(offset[1]))
        Dataset.DatasetAddOffsetZ(builder, float(offset[2]))
        Dataset.DatasetAddLmax(builder, int(QLevel))
        Dataset.DatasetAddName(builder, name_str)

        dataset = Dataset.DatasetEnd(builder)
        builder.Finish(dataset)

        # Save to file
        buf = builder.Output()
        out_path = os.path.join(saveMatDir, ptName + '.fb')
        with open(out_path, 'wb') as f:
            f.write(buf)

        # print(f"Saved FlatBuffer to {out_path}")

        t_save = time.time()
        # print(f"  [Profile] FlatBuffer Save:   {t_save - t_ctx:.4f}s")

    except ImportError as e:
        print(f"Skipping FlatBuffer generation: {e}")
        print("Please ensure 'flatbuffers' is installed and 'schema.fbs' is compiled.")

        # Fallback to .mat if needed, or just warn.
        # User requested move to flatbuffers, so we should prioritize that.
        # But keeping .mat for now might be safe if user wants to debug.
        # Uncomment below to keep .mat
        # hdf5storage.savemat(os.path.join(saveMatDir, ptName + '.mat'), patchFile, format='7.3', oned_as='row', store_python_metadata=True)

    DQpt = (pt * qs + offset)
    return os.path.join(saveMatDir, ptName + '.fb'), DQpt, refPt
