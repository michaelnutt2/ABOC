"""
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
"""
import datetime
import os

import pt as pointCloud
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload, CPrintl, expName, device
from octAttention import model

# ############# warning ###############
# # decoder.py and test.py rely on this model here
# # do not move this lines to somewhere else
model = model.to(device)
saveDic = reload(None, 'modelsave/lidar/encoder_epoch_00801460.pth', multiGPU=False)
model.load_state_dict(saveDic['encoder'])

# ##########LiDar##############
GPCC_MULTIPLE = 2 ** 20
list_orifile = ['file/Ply/11_000000.bin']
if __name__ == "__main__":
    printl = CPrintl(expName + '/encoderPLY.txt')
    printl('_' * 50, 'OctAttention V0.4', '_' * 50)
    printl(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    printl('load checkpoint', saveDic['path'])
    for oriFile in list_orifile:
        printl(oriFile)
        ptName = os.path.splitext(os.path.basename(oriFile))[0]
        # Generate FlatBuffer
        fbFile, DQpt, normalizePt = dataPrepare(oriFile, saveMatDir='./Data/testPly', offset='min',
                                                qs=2 / (2 ** 12 - 1), rotation=False, normalize=True)

        # Load FlatBuffer and Prepare Data
        # We can use dataset.DataFolder logic or direct load here.
        # Direct load using generated classes is best for transparency.
        try:
            import OctreeData.Dataset as Dataset
            # import OctreeData.OctreeNode as OctreeNode # Not needed here maybe

            with open(fbFile, 'rb') as f:
                buf = f.read()
                dataset = Dataset.Dataset.GetRootAsDataset(buf, 0)

            nodes_len = dataset.NodesLength()
            context_len = 17

            # Extract Features [N, 17, 3] and Targets [N]
            features = np.zeros((nodes_len, context_len, 3), dtype=np.int32)
            targets = np.zeros(nodes_len, dtype=np.int32)

            for i in range(nodes_len):
                node = dataset.Nodes(i)
                parent_occ = node.ParentOccupancy()
                neighbors_len = node.NeighborOccupanciesLength()
                neighbors = np.zeros(neighbors_len, dtype=int)
                for j in range(neighbors_len):
                    neighbors[j] = node.NeighborOccupancies(j)

                full_ctx = np.zeros(context_len, dtype=int)
                full_ctx[0:8] = neighbors[0:8]
                full_ctx[8] = parent_occ
                full_ctx[9:] = neighbors[8:]

                lvl = node.Level()
                octant = node.Octant()
                occ = node.Occupancy()

                features[i, :, 0] = full_ctx
                features[i, :, 1] = lvl - 1
                features[i, :, 2] = octant # Wait, encoderTool previously unused octant?
                # Model needs Octant in Channel 2.
                # In dataset.py we put 0. Here we put octant.
                # Actually, in 'dataset.py' update I put: extracted_data[i, 8, 2] = node.Octant()
                # and zeros elsewhere.
                # The Transformer takes [17, N, 3].
                # It uses index 2 (Octant) via 'encoder2'.
                # If we only set center octant, others are 0 (Padding).
                # Is that what we want?
                # "Query (Parent + Child Index)". Yes, only the center matters for the Query.
                # The context neighbors are "Result of previous decodes", their 'Child Index' is irrelevant to the current prediction?
                # Actually, neighbors are just "Context". They don't have a "Child Index" relative to *Current* parent.
                # So setting their octant to 0 is fine.

                features[i, 8, 2] = octant
                targets[i] = occ

            data_packet = {
                'features': features,
                'targets': targets,
                'ptName': ptName,
                'visualize_data': (normalizePt, DQpt) # For error check
            }

            main(data_packet, model, actualcode=True, printl=printl)

            print('_' * 50, 'pc_error', '_' * 50)
            pointCloud.pcerror(normalizePt, DQpt, None, '-r 1', None).wait()

        except ImportError:
            print("Error: Could not import FlatBuffer classes. Run compile_schema.sh")
