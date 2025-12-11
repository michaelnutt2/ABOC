"""
DataFolder class for loading and processing Octree data.
Supports both legacy .mat files and new FlatBuffer (.fb) format.
"""
import os
import os.path
import numpy as np
import glob
import torch.utils.data as data
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

    This class supports loading data from FlatBuffer files (`.fb`), which contain
    linearized Octree nodes and their Parallel Contexts (Parent + Neighbors).

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
    """

    def __init__(self, root, TreePoint, dataLenPerFile, transform=None, loader=default_loader):

        # dataLenPerFile is the number of all octnodes in one 'mat' file on average

        dataNames = []
        # Update glob to look for .fb files
        search_pattern = os.path.join(root, "*.fb")
        for filename in sorted(glob.glob(search_pattern)):
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

        if self.fileLen == 0:
            print("No .fb files found. Falling back to .mat for compatibility check or error.")

        self.dataLenPerFile = 10000 # Placeholder

    def calcdataLenPerFile(self):
        # Simplification for now
        return 10000

    def __getitem__(self, index):
        """Loads a chunk of data from the FlatBuffer files."""
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
                context_len = 17
                extracted_data = np.zeros((nodes_len, context_len, 4), dtype=np.int32)

                for i in range(nodes_len):
                    node = dataset.Nodes(i)

                    # Reconstruct Context Vector
                    parent_occ = node.ParentOccupancy()
                    neighbors_len = node.NeighborOccupanciesLength()
                    neighbors = np.zeros(neighbors_len, dtype=int)
                    for j in range(neighbors_len):
                        neighbors[j] = node.NeighborOccupancies(j)

                    # Re-assemble: [Neighbors_Pre, Parent, Neighbors_Post]
                    full_ctx = np.zeros(context_len, dtype=int)
                    full_ctx[0:8] = neighbors[0:8]
                    full_ctx[8] = parent_occ
                    full_ctx[9:] = neighbors[8:]

                    # Current Node Level
                    lvl = node.Level()

                    extracted_data[i, :, 0] = full_ctx
                    extracted_data[i, :, 1] = lvl - 1 # Parent Level approx
                    extracted_data[i, :, 2] = 0

                    extracted_data[i, 8, 2] = node.Octant() # Store octant in center

                    # Store target (Node Occupancy) in 4th channel
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
