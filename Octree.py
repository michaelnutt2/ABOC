"""
Author: fuchy@stu.pku.edu.cn
Description: Octree
FilePath: /compression/Octree.py
All rights reserved.
"""
import numpy as np
from OctreeCPP.Octreewarpper import GenOctree


class CNode:
    def __init__(self, nodeid=0, childPoint=[[]] * 8, parent=0, oct=0, pos=np.array([0, 0, 0]), octant=0) -> None:
        self.nodeid = nodeid
        self.childPoint = childPoint.copy()
        self.parent = parent
        self.oct = oct  # occupancyCode 1~255
        self.pos = pos
        self.octant = octant  # 1~8


class COctree:
    def __init__(self, node=[], level=0) -> None:
        self.node = node.copy()
        self.level = level


def dec2bin(n, count=8):
    """returns the binary of integer n, using count number of digits"""
    return [int((n >> y) & 1) for y in range(count - 1, -1, -1)]


def dec2binAry(x, bits):
    mask = np.expand_dims(2 ** np.arange(bits - 1, -1, -1), 1).T
    return (np.bitwise_and(np.expand_dims(x, 1), mask) != 0).astype(int)


def bin2decAry(x):
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    bits = x.shape[1]
    mask = np.expand_dims(2 ** np.arange(bits - 1, -1, -1), 1)
    return x.dot(mask).astype(int)


def Morton(A):
    A = A.astype(int)
    n = np.ceil(np.log2(np.max(A) + 1)).astype(int)
    x = dec2binAry(A[:, 0], n)
    y = dec2binAry(A[:, 1], n)
    z = dec2binAry(A[:, 2], n)
    m = np.stack((x, y, z), 2)
    m = np.transpose(m, (0, 2, 1))
    mcode = np.reshape(m, (A.shape[0], 3 * n), order='F')
    return mcode


def DeOctree(Codes):
    Codes = np.squeeze(Codes)
    occupancyCode = np.flip(dec2binAry(Codes, 8), axis=1)
    codeL = occupancyCode.shape[0]
    N = np.ones(30, int)
    codcal = 0
    L = 0
    while codcal + N[L] <= codeL:
        L += 1
        try:
            N[L + 1] = np.sum(occupancyCode[codcal:codcal + N[L], :])
        except:
            assert 0
        codcal = codcal + N[L]
    Lmax = L
    Octree = [COctree() for _ in range(Lmax + 1)]
    proot = [np.array([0, 0, 0])]
    Octree[0].node = proot
    codei = 0
    for L in range(1, Lmax + 1):
        childNode = []  # the node of next level
        for currentNode in Octree[L - 1].node:  # bbox of currentNode
            code = occupancyCode[codei, :]
            for bit in np.where(code == 1)[0].tolist():
                newnode = currentNode + (np.array(dec2bin(bit, count=3)) << (Lmax - L))  # bbox of childnode
                childNode.append(newnode)
            codei += 1
        Octree[L].node = childNode.copy()
    points = np.array(Octree[Lmax].node)
    return points


def GenKparentSeq(Octree, K):
    LevelNum = len(Octree)
    nodeNum = Octree[-1].node[-1].nodeid
    Seq = np.ones((nodeNum, K), 'int') * 255
    LevelOctant = np.zeros((nodeNum, K, 2), 'int')  # Level and Octant
    Pos = np.zeros((nodeNum, K, 3), 'int')  # padding 0
    ChildID = [[] for _ in range(nodeNum)]
    Seq[0, K - 1] = Octree[0].node[0].oct
    LevelOctant[0, K - 1, 0] = 1
    LevelOctant[0, K - 1, 1] = 1
    Pos[0, K - 1, :] = Octree[0].node[0].pos
    Octree[0].node[0].parent = 1  # set to 1
    n = 0
    for L in range(0, LevelNum):
        for node in Octree[L].node:
            Seq[n, K - 1] = node.oct
            Seq[n, 0:K - 1] = Seq[node.parent - 1, 1:K]
            LevelOctant[n, K - 1, :] = [L + 1, node.octant]
            LevelOctant[n, 0:K - 1] = LevelOctant[node.parent - 1, 1:K, :]
            Pos[n, K - 1] = node.pos
            Pos[n, 0:K - 1, :] = Pos[node.parent - 1, 1:K, :]
            if L == LevelNum - 1:
                pass
            n += 1
    assert n == nodeNum
    DataStruct = {'Seq': Seq, 'Level': LevelOctant, 'ChildID': ChildID, 'Pos': Pos}
    return DataStruct


def GenParallelContext(Octree, context_range=8):
    # Find max nodeid to size arrays
    nodeNum = 0
    if Octree and Octree[-1].node:
        nodeNum = Octree[-1].node[-1].nodeid

    # Flatten oct codes for easy access
    # Using nodeid as index. Note: nodeid is likely 1-based (root=1).
    # We allocate nodeNum + 1.
    AllOcts = np.zeros(nodeNum + 1, dtype=int)

    # Fill AllOcts
    for level in Octree:
        for node in level.node:
             if node.nodeid <= nodeNum:
                 AllOcts[node.nodeid] = node.oct

    # Output context array
    # Shape: (nodeNum, 2 * context_range + 1)
    # Format: [Parent, P-1, P-2, ... P-8, P+1, ... P+8] -> actually let's sort strictly:
    # [P-range, ..., P, ..., P+range] so convolution is natural.
    width = 2 * context_range + 1
    Context = np.zeros((nodeNum, width), dtype=int)

    # Iterate to fill context
    # Note: GenKparentSeq iterates n=0..nodeNum-1 matching the sequence.
    # We should match the order of nodes in the output sequence.
    # GenKparentSeq iterates levels then nodes.

    n = 0
    for L in range(len(Octree)):
        for node in Octree[L].node:
            # Current node corresponds to row n in the dataset
            # We need context for this node.
            # Context is based on Parent's neighbors.

            p_id = node.parent

            # Root (L=0) handling: parent is 1 (itself) in GenKparentSeq.
            # Neighbors of root? None.

            if p_id > 0 and p_id <= nodeNum:
                # Get window around parent

                # Determine range [start, end] inclusive
                start = max(1, p_id - context_range)
                end = min(nodeNum, p_id + context_range)

                # We need to map [start, end] into the Context row [0, width]
                # The center of Context row is 'context_range', which should hold AllOcts[p_id].
                # A value at global index 'k' should go to Context column: context_range + (k - p_id)

                val_slice = AllOcts[start : end+1]

                col_start = context_range + (start - p_id)
                col_end = context_range + (end - p_id) + 1

                Context[n, col_start : col_end] = val_slice

            n += 1

    return Context, AllOcts



if __name__ == '__main__':
    import pt

    p = pt.ptread('pt0.ply')
    print(p.max())
    code, Octree, QLevel = GenOctree(p)
    print(len(code))
    dp = DeOctree(code)
    pt.pcerror(p, dp, None, '-r 1024', None).wait()
