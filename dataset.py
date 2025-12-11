import os
import os.path
import numpy as np
import glob
import torch.utils.data as data
from PIL import Image
import glob
# import scipy.io as scio
import h5py
from networkTool import trainDataRoot, levelNumK

IMG_EXTENSIONS = [
    'Kitti'
]


def is_image_file(filename):
    return any(extension in filename for extension in IMG_EXTENSIONS)


def default_loader(path):
    # print(path)
    mat = h5py.File(path)
    # data = scio.loadmat(path)
    cell = mat['patchFile']
    return cell, mat


class DataFolder(data.Dataset):
    """DataFolder is a custom Dataset class for handling octree data from files.

    This class is designed to load and process octree data stored in .mat files,
    providing batch-wise access to tree point data.

    Args:
        root (str): Root directory path containing the data files
        TreePoint (int): Number of tree points to process in each batch
        dataLenPerFile (int): Average number of octnodes in one .mat file
        transform (callable, optional): Optional transform to be applied on a sample. Defaults to None
        loader (callable, optional): Function to load the data. Defaults to default_loader

    Attributes:
        root (str): Root directory path
        dataNames (list): Sorted list of data file paths
        transform (callable): Transform to be applied to the data
        loader (callable): Function used to load the data
        index (int): Current index in the data buffer
        datalen (int): Length of current data in buffer
        dataBuffer (list): Buffer holding loaded data
        fileIndx (int): Current file index being processed
        TreePoint (int): Number of tree points per batch
        fileLen (int): Total number of files
        dataLenPerFile (int): Average number of octnodes per file

    Methods:
        calcdataLenPerFile():
            Calculates the average number of octnodes per file
            Returns:
                float: Average number of octnodes per file

        __getitem__(index):
            Gets item at specified index
            Args:
                index (int): Index of the item to get
            Returns:
                list: Processed data batch

        __len__():
            Returns total number of batches
            Returns:
                int: Total number of available batches
    """

    def __init__(self, root, TreePoint, dataLenPerFile, transform=None, loader=default_loader):

        # dataLenPerFile is the number of all octnodes in one 'mat' file on average

        dataNames = []
        # Update glob to look for .fb files
        search_pattern = os.path.join(root, "*.fb")
        for filename in sorted(glob.glob(search_pattern)):
            # if is_image_file(filename): # Skipped check for verify
            dataNames.append('{}'.format(filename))

        self.root = root
        self.dataNames = sorted(dataNames)
        self.transform = transform
        self.loader = loader
        self.index = 0
        self.datalen = 0
        self.dataBuffer = []
        self.fileIndx = 0
        self.TreePoint = TreePoint
        self.fileLen = len(self.dataNames)
        # assert self.fileLen > 0, 'no file found!'
        if self.fileLen == 0:
            print("No .fb files found. Falling back to .mat for compatibility check or error.")

        # self.dataLenPerFile = dataLenPerFile
        # self.dataLenPerFile = self.calcdataLenPerFile()
        self.dataLenPerFile = 10000 # Placeholder or calc

    def calcdataLenPerFile(self):
        # Simplification for now
        return 10000

    def __getitem__(self, index):
        import OctreeData.Dataset as Dataset
        import OctreeData.OctreeNode as OctreeNode

        while self.index + self.TreePoint > self.datalen:
            filename = self.dataNames[self.fileIndx]
            if self.dataBuffer:
                a = [self.dataBuffer[0][self.index:].copy()]
            else:
                a = []

            # Load FlatBuffer
            with open(filename, 'rb') as f:
                buf = f.read()
                dataset = Dataset.Dataset.GetRootAsDataset(buf, 0)

                # Extract nodes
                nodes_len = dataset.NodesLength()

                # We want to construct an array of shape [nodes_len, 17, 3]
                # 17 = Context Range (Parent + Neighbors)
                # 3 = [Occupancy, Level, Octant]

                # Pre-allocate numpy array
                context_len = 17 # 8 + 1 + 8
                # We need to read the context from FB

                # Iterate nodes
                # Note: FlatBuffers random access is fast.

                # To optimize, we might want to process in chunks or list comprehension
                # But treating FB as array is tricky.

                # Let's create a list of processed nodes
                extracted_data = np.zeros((nodes_len, context_len, 3), dtype=np.int32)

                for i in range(nodes_len):
                    node = dataset.Nodes(i)

                    # Reconstruct Context Vector
                    # Schema: parent_occupancy (int), neighbor_occupancies ([int])
                    # We want [Neighbors_Pre, Parent, Neighbors_Post]?
                    # Or just the sequence as stored.
                    # dataPrepare stored: [Neighbors...].
                    # Wait, dataPrepare split them: parent_occ and neighbor_vec.
                    # We should reconstruct the full context sequence.
                    # In dataPrepare: we removed parent from neighbors list.
                    # Neighbors list was reversed? "for val in reversed(neighbors): builder.PrependInt32(val)"
                    # Prepend reverses order, so they are stored in original order in the buffer?
                    # Let's assume standard order.

                    parent_occ = node.ParentOccupancy()
                    neighbors_len = node.NeighborOccupanciesLength()
                    neighbors = np.zeros(neighbors_len, dtype=int)
                    for j in range(neighbors_len):
                        neighbors[j] = node.NeighborOccupancies(j)

                    # Re-assemble: [Neighbors_Pre, Parent, Neighbors_Post]
                    # We don't know exactly where Parent was, but assuming centered context_range=8.
                    # dataPrepare: neighbors = np.delete(ctx, 8).
                    # So center of original ctx was 8.
                    # neighbors has 16 elements. 0-7 are Pre, 8-15 are Post.

                    full_ctx = np.zeros(context_len, dtype=int)
                    full_ctx[0:8] = neighbors[0:8]
                    full_ctx[8] = parent_occ
                    full_ctx[9:] = neighbors[8:]

                    # Features: [Occupancy, Level, Octant]
                    # Only Occupancy varies across context. Level/Octant are from *Parent*?
                    # The Context is "Parent's Neighbors".
                    # They share the same Level (Parent's Level).
                    # Their Octant? They are siblings, so they have octants.
                    # But we are just passing Occupancy codes.
                    # The 'dataset.py' returns 'data' which is used as 'src'.
                    # 'src' expects [..., 3] usually.
                    # Let's fill:
                    # dim 0: Occupancy (Context)
                    # dim 1: Level (Node Level - 1 approx? Or Parent Level)
                    # dim 2: Octant (Neighbor Index? Or 0)

                    # Current Node Level
                    lvl = node.Level()

                    extracted_data[i, :, 0] = full_ctx
                    extracted_data[i, :, 1] = lvl - 1 # Parent Level approx
                    extracted_data[i, :, 2] = 0 # Placeholder for octant/pos encoding

                    # Append Target?
                    # "get_batch" in octAttention splits data/target.
                    # Target is "Next Timestep" in original code.
                    # But here Target is "Current Node Occupancy".
                    # We should append Target to the features or handle it.
                    # Original: data has "Kparent" sequence.
                    # Target was derived from source[i+1].
                    # This implies Source[i] predicts Source[i+1].

                    # NEW PARADIGM:
                    # We want to predict Node[i].Occupancy using Node[i].Context.
                    # So we should package them together.
                    # Let's add Target as a 4th feature? Or separate output?
                    # The DataFolder structure is rigid (returns one array).
                    # Original returned: [ptNum, Kparent, ...].
                    # Let's return: [nodes_len, context_len, 4]
                    # 4th dim = Target (Node.Occupancy).

                    extracted_data[i, 8, 2] = node.Octant() # Store octant in center?
                    # Store target (Node Occupancy) in a special slot?
                    # Actually, let's look at get_batch.
                    # It expects a tensor.
                    # We can store Target in data[:, :, 0] of a "Next" step?
                    # Or just return [nodes_len, context_len, 4].

                    # Let's use 4 channels: [CtxOcc, Level, Octant, Target]
                    extracted_data[i, :, 3] = node.Occupancy()

                a.append(extracted_data) # [N, 17, 4]

            self.dataBuffer = []
            self.dataBuffer.append(np.vstack(tuple(a)))

            self.datalen = self.dataBuffer[0].shape[0]
            self.fileIndx += 1
            self.index = 0
            if self.fileIndx >= self.fileLen:
                self.fileIndx = index % self.fileLen

        # Return a chunk
        img = []
        chunk = self.dataBuffer[0][self.index:self.index + self.TreePoint]
        img.append(chunk)

        self.index += self.TreePoint

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return int(1000 * self.fileLen) # Placeholder



if __name__ == "__main__":
    file_name = "/home/michaelnutt/Projects/OctAttention/Data/Lidar/train/Kitti_00000000.mat"
    mat = h5py.File(file_name)
    # print(mat)
    cell = mat['patchFile']
    # print(cell)
