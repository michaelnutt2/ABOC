"""
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-20 08:06:11
LastEditTime: 2021-09-20 23:53:24
LastEditors: fcy
Description: the training file
             see networkTool.py to set up the parameters
             will generate training log file loss.log and checkpoint in folder 'expName'
FilePath: /compression/octAttention.py
All rights reserved.
"""
import math
import torch
import torch.nn as nn
import os
import datetime
from networkTool import *
from torch.utils.tensorboard import SummaryWriter
from attentionModel import TransformerLayer, TransformerModule

##########################

ntokens = 255  # the size of vocabulary
ninp = 128 + 4 + 6  # embedding dimension (138)
nhid = 300  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # the number of heads in the multiheadattention models
dropout = 0  # the dropout value
batchSize = 32


class TransformerModel(nn.Module):
    """
    A Transformer model for processing octree data.

    This model implements a transformer architecture specifically designed for octree representations,
    with support for handling octant positions, levels, and features.

    Args:
        ntoken (int): Size of the token vocabulary
        ninp (int): Size of input embeddings/features
        nhead (int): Number of attention heads
        nhid (int): Dimension of feedforward network model
        nlayers (int): Number of transformer layers
        dropout (float, optional): Dropout value between 0 and 1. Defaults to 0.5

    Attributes:
        model_type (str): Type identifier for the model ('Transformer')
        pos_encoder (PositionalEncoding): Positional encoding layer
        transformer_encoder (TransformerModule): Main transformer encoder stack
        encoder (nn.Embedding): Token embedding layer
        encoder1 (nn.Embedding): Level embedding layer
        encoder2 (nn.Embedding): Octant embedding layer
        ninp (int): Size of input embeddings
        act (nn.ReLU): Activation function
        decoder0 (nn.Linear): First decoder layer
        decoder1 (nn.Linear): Second decoder layer

    Methods:
        generate_square_subsequent_mask(sz): Generates attention mask for self-attention
        init_weights(): Initializes model weights using Xavier initialization
        forward(src, src_mask, dataFeat): Forward pass of the model

    The forward pass processes input tensors containing:
        - Octree values (0-254)
        - Level information (0-12)
        - Octant positions (0-8)
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 128)
        self.encoder1 = nn.Embedding(MAX_OCTREE_LEVEL + 1, 6)
        self.encoder2 = nn.Embedding(9, 4)

        print(self.encoder)
        print("Ninp:", ninp)
        print("ntok", ntoken)

        self.ninp = ninp
        self.act = nn.ReLU()
        self.decoder0 = nn.Linear(ninp, ninp)
        self.decoder1 = nn.Linear(ninp, ntoken)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        self.encoder.weight.data = nn.init.xavier_normal_(self.encoder.weight.data)
        self.decoder0.bias.data.zero_()
        self.decoder0.weight.data = nn.init.xavier_normal_(self.decoder0.weight.data)
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data = nn.init.xavier_normal_(self.decoder1.weight.data)

    def forward(self, src, src_mask, dataFeat):
        bptt = src.shape[0] # Should be 17
        batch = src.shape[1]

        oct = src[:, :, 0]  # [17, N]
        level = src[:, :, 1] # [17, N]
        octant = src[:, :, 2] # [17, N]

        # level -= torch.clip(level[:, :, -1:] - 12, 0, None)
        # Level correction might fail if shapes changed.
        # Level is now full [17, N].
        # Let's assume input is correct or clamp it.
        torch.clip_(level, 0, MAX_OCTREE_LEVEL)

        aOct = self.encoder(oct.long())
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct, aLevel, aOctant), 2) # dim 2 was 3 (feature dim).
        # Shape of encoders: [17, N, Embed]
        # cat dim 2 -> [17, N, TotalEmbed]
        # Original: a = torch.cat((...), 3). Reshape.
        # Original src had 4 dimensions. New src has 3 dimensions [Seq, Batch, 3].

        # a is now [17, N, Hidden]. matches self.ninp?
        # ninp = 4 * (128 + 4 + 6)?
        # Encoders: 128, 6, 4. Sum = 138.
        # ninp defined as 4 * sum? NO.
        # Original: cat dimension 3.
        # src was [bptt, batch, levels, feat_idx].
        # levels=4.
        # It concatenated embeddings of *all levels*?
        # "a = torch.cat((aOct, aLevel, aOctant), 3)" -> [bptt, batch, levels, sum_embed].
        # "a.reshape((bptt, batch, -1))" -> flattened levels.
        # So input feature vector was concatenation of 4 levels' embeddings.

        # New Logic:
        # We have 17 neighbors.
        # We process them as a SEQUENCE (Transformers love sequences).
        # We do NOT flatten them.
        # We treat each neighbor as a token.
        # Token Dimension = 128 + 6 + 4 = 138.
        # self.ninp should be 138.
        # Current self.ninp = 4 * 138 (because of 4 levels previously).

        # Issue: Model parameters (self.decoder0 etc) depend on ninp.
        # If I change ninp, I change the model architecture.
        # The user said "build on it... update model training".
        # If the input dimension changes heavily, we must adjust ninp.
        # Since we use sequence of 17, and not concatenated 4, our embedding dim corresponds to 1 neighbor.

        # FIX:
        # We need to project 'a' to 'ninp'.
        # Or change 'ninp' definition in the script.
        # self.ninp is defined globally in 'octAttention.py' (ninp = ...).
        # I should change that definition too if I want to match.
        # Or I project [17, N, 138] to [17, N, 552] (padding/linear)?
        # Better to change 'ninp' to 138.

        src = a * math.sqrt(src.shape[-1]) # Scale by embedding dim

        # Project to model dimension if needed?
        # TransformerEncoder expects d_model.
        # If we change ninp, we change d_model.

        # Assume ninp is updated to ~138.

        # output = self.transformer_encoder(src, src_mask)
        # return output

        # BUT:
        # If I don't update 'ninp' in the global scope, the model init will fail or shape mismatch.
        # I cannot edit the global variable easily with replace_content unless I target it.
        # I will target it.

        return self.decoder1(self.act(self.decoder0(self.transformer_encoder(src, src_mask))))


######################################################################
# ``PositionalEncoding`` module
#

class PositionalEncoding(nn.Module):
    """Positional Encoding module for transformer models.

    This class implements the positional encoding described in 'Attention Is All You Need'
    (Vaswanov et al., 2017). It adds positional information to the input embeddings using
    sine and cosine functions of different frequencies.

    Args:
        d_model (int): The dimension of the model's embeddings
        dropout (float, optional): Dropout rate. Defaults to 0.1
        max_len (int, optional): Maximum sequence length. Defaults to 5000

    Attributes:
        dropout (nn.Dropout): Dropout layer
        pe (Tensor): Positional encoding matrix of shape (max_len, 1, d_model)

    Example:
        >>> pos_encoder = PositionalEncoding(512, 0.1)
        >>> x = torch.randn(10, 32, 512)  # (seq_len, batch_size, d_model)
        >>> encoded = pos_encoder(x)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


######################################################################
# Functions to generate input and target sequence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def get_batch(source, i):
    """
    Prepares a batch of data and target for training or evaluation.

    This function generates a batch of sequences from the source data by sliding a window
    over the input sequence. It handles the temporal arrangement of features and targets
    for octant-based analysis.
    Args:
        source (torch.Tensor): The source data tensor with shape
            [sequence_length, batch_size, num_levels, features].
        i (int): Starting index in the source sequence.
    Returns:
        tuple: A tuple containing:
            - data (torch.Tensor): Processed batch data containing only the last K levels,
              where features are arranged such that current node features are in the last row.
            - target (torch.Tensor): Target values for prediction, containing octant values
              for the next timestep, reshaped to 1D tensor.
            - empty_list (list): An empty list placeholder for additional batch information.
    Notes:
        - The function uses a global 'bptt' (backpropagation through time) variable to
          determine sequence length.
        - 'levelNumK' is a global variable determining how many levels to include in output.
        - The function rearranges features so that each timestep can access the next
          timestep's features (except octant) as known information.
    """

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len].clone()
    target = source[i + 1:i + 1 + seq_len, :, -1, 0].reshape(-1)
    data[:, :, 0:-1, :] = source[i + 1:i + seq_len + 1, :, 0:-1, :]  # this moves the feat(octant,level) of current node
    # to last row,
    data[:, :, -1, 1:3] = source[i + 1:i + seq_len + 1, :, -1, 1:3]  # which will be used as known feat
    return data[:, :, -levelNumK:, :], target.long(), []


######################################################################
# Run the model
# -------------
#
model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)
if __name__ == "__main__":
    import dataset
    import torch.utils.data as data
    import time
    import os


    epochs = 8  # The number of epochs
    best_model = None
    batch_size = 128
    TreePoint = bptt * 16
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint, transform=None,
                                   dataLenPerFile=None)  # you should run 'dataLenPerFile' in dataset.py to get this
    # num (17456051.4)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)  # will load TreePoint*batch_size at one time

    # loger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName + '/loss.log')
    writer = SummaryWriter('./log/' + expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    # model_structure(model,printl)
    printl(expComment + ' Pid: ' + str(os.getpid()))
    log_interval = int(batch_size * TreePoint / batchSize / bptt)

    # learning
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # reload
    saveDic = None
    # saveDic = reload(100030,checkpointPath)
    if saveDic:
        scheduler.last_epoch = saveDic['epoch'] - 1
        idloss = saveDic['idloss']
        best_val_loss = saveDic['best_val_loss']
        model.load_state_dict(saveDic['encoder'])


    def get_batch(source, i):
        # Input source: [seq_len(bptt), batch_size, 17, 4]
        # We process 'bptt' nodes at comparable time.
        # But actually, each node is independent now.
        # Source contains [Ctx, Level, Octant, Target].

        # We need to return:
        # data: [17, bptt * batch_size, 3] (Input features)
        # target: [bptt * batch_size]

        seq_len = min(bptt, len(source) - i)

        # Slice current window
        chunk = source[i:i + seq_len] # [actual_seq, batch, 17, 4]

        # Flatten (actual_seq * batch) -> Total Batch
        flat_chunk = chunk.reshape(-1, 17, 4) # [N, 17, 4]

        # Extract features and target
        # Features: [:, :, 0:3] -> [N, 17, 3]
        # Target: [:, 8, 3] -> Center node's target is in 3rd channel.
        # Note: In dataset.py we put Target in the last channel.
        # All 17 neighbors have the same Target in their row?
        # In dataset.py: extracted_data[i, :, 3] = node.Occupancy()
        # Yes, repeated. So take any. Center index 8 is safe.

        targets = flat_chunk[:, 8, 3].long()
        data = flat_chunk[:, :, 0:3] # [N, 17, 3]

        # Transformer expects [Seq, Batch, Feats]
        data = data.permute(1, 0, 2) # [17, N, 3]

        return data, targets, []


    def train(epoch):
        print("--Starting Training--")
        global idloss, best_val_loss
        model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        total_loss_list = torch.zeros((1, 7))

        for Batch, d in enumerate(train_loader):
            print("--Training Batch " + str(Batch) + "--")

            # d[0] shape: [batchSize * (?) , ?] from DataLoader
            # DataLoader loads TreePoint * batch_size items.
            # But we reshaped d[0] in original code.
            # New DataLoader returns [Batch, TreePoint, 17, 4] if batch_size > 1?
            # DataLoader with batch_size=N stacks them.
            # d[0] is [batch_size, TreePoint, 17, 4].

            # Reshape to [TreePoint, batch_size, 17, 4] to mimic "sequence" logic or just flatten
            train_data = d[0].permute(1, 0, 2, 3).to(device) # [TreePoint, batch_size, 17, 4]

            # No Mask needed (or mask neighbors?)
            # All neighbors are visible.
            src_mask = None

            for index, i in enumerate(range(0, train_data.size(0), bptt)):
                # print("--In a loop at index " + str(index) + "--")
                data, targets, dataFeat = get_batch(train_data, i)

                optimizer.zero_grad()

                # Model forward
                # data: [17, N, 3]
                output = model(data, src_mask, dataFeat)  # output: [17, N, 255]?

                # We only want prediction for the Center node?
                # Or do we aggregate the Transformer output?
                # Usually standard Transfomer predicts "Next Token" or "Masked Token".
                # But here we are classifying the Center Node based on Context.
                # If we feed [17, N, 3]. Output is [17, N, 255].
                # Which token output corresponds to the classification?
                # Approaches:
                # 1. Take output at Center Index (8)?
                # 2. Take output at last index?
                # 3. Average pool?
                # The "Decoder1" projects to 255 (classes).
                # Let's assume we use the output at the center position (8) to predict the center node?
                # Or maybe position 0 (CLS token equivalent)?
                # Given we trained it as "Sequence", probably the center corresponds to the parent?
                # The relationship "Context -> Child" is learned.
                # If we use the output corresponding to the Parent token (index 8) to predict the Child?
                # Let's try Center Index (8).

                output = output[8, :, :] # [N, 255]

                loss = criterion(output, targets) / math.log(2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()

                batch = index # Reuse variable name logic
                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time

                    total_loss_list = " - "
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                           'lr {:02.2f} | ms/batch {:5.2f} | '
                           'loss {:5.2f} | losslist  {} | ppl {:8.2f}'.format(
                        epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                                             elapsed * 1000 / log_interval,
                        cur_loss, total_loss_list, math.exp(cur_loss)))
                    total_loss = 0

                    start_time = time.time()

                    writer.add_scalar('train_loss', cur_loss, idloss)
                    idloss += 1

            if Batch % 10 == 0:
                save(epoch * 100000 + Batch, saveDict={'encoder': model.state_dict(), 'idloss': idloss, 'epoch': epoch,
                                                       'best_val_loss': best_val_loss}, modelDir=checkpointPath)


    # train
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
