"""
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-20 08:06:11
LastEditTime: 2025-12-11
LastEditors: Antigravity
Description: OctAttention Training Script.
             Implements training loop for the Parallel Context Octree Model.
             See networkTool.py to set up the parameters.
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

ntokens = 256  # the size of vocabulary (0-255)
ninp = 130 + 4 + 6  # embedding dimension (140, divisible by 4)
nhid = 300  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 3  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
epochs = 1  # 1 Epoch for testing
nhead = 4  # the number of heads in the multiheadattention models
dropout = 0  # the dropout value
batchSize = 32


class TransformerModel(nn.Module):
    """
    A Transformer model for processing parallel octree contexts.

    The model takes a context sequence (Parent + Neighbors) and predicts the occupancy
    of the current node.

    Args:
        ntoken (int): Size of the token vocabulary
        ninp (int): Size of input embeddings/features
        nhead (int): Number of attention heads
        nhid (int): Dimension of feedforward network model
        nlayers (int): Number of transformer layers
        dropout (float, optional): Dropout value. Defaults to 0.5
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerModule(encoder_layers, nlayers)

        self.encoder = nn.Embedding(ntoken, 130)  # Adjusted to 130
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
        """
        Forward pass of the model.

        Args:
            src: Input tensor of shape [17, Batch, 3].
                 Channels: 0:Occupancy, 1:Level, 2:Octant.
            src_mask: Optional attention mask.
            dataFeat: Unused/Legacy feature input.

        Returns:
            Output tensor of shape [..., ntokens].
        """
        bptt = src.shape[0] # Should be 17
        batch = src.shape[1]

        oct = src[:, :, 0]  # [17, N]
        level = src[:, :, 1] # [17, N]
        octant = src[:, :, 2] # [17, N]

        torch.clip_(level, 0, MAX_OCTREE_LEVEL)

        aOct = self.encoder(oct.long())
        aLevel = self.encoder1(level.long())
        aOctant = self.encoder2(octant.long())

        a = torch.cat((aOct, aLevel, aOctant), 2)

        # New Logic: Treat neighbors as tokens in a sequence.
        # Total Token Dimension = 128 + 6 + 4 = 138.

        src = a * math.sqrt(src.shape[-1]) # Scale by embedding dim

        return self.decoder1(self.act(self.decoder0(self.transformer_encoder(src, src_mask))))


######################################################################
# ``PositionalEncoding`` module
#

class PositionalEncoding(nn.Module):
    """Positional Encoding module for transformer models."""

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
    Extracts a batch of features and targets from the source data.

    Args:
        source (torch.Tensor): Input data chunk [SeqLen, Batch, 17, 4].
        i (int): Current index in the sequence.

    Returns:
        data: Input features [17, BatchSize, 3].
        targets: Target occupancies [BatchSize].
        dataFeat: Empty list (unused).
    """

    seq_len = min(bptt, len(source) - i)

    # Slice current window
    chunk = source[i:i + seq_len] # [actual_seq, batch, 17, 4]

    # Flatten (actual_seq * batch) -> Total Batch
    flat_chunk = chunk.reshape(-1, 17, 4) # [N, 17, 4]

    # Extract features and target
    # Features: [:, :, 0:3] -> [N, 17, 3]
    # Target: [:, 8, 3] -> Center node's target is in 3rd channel.

    targets = flat_chunk[:, 8, 3].long()
    data = flat_chunk[:, :, 0:3] # [N, 17, 3]

    # Transformer expects [Seq, Batch, Feats]
    data = data.permute(1, 0, 2) # [17, N, 3]

    return data, targets, []



model = TransformerModel(ntokens, ninp, nhead, nhid, nlayers, dropout).to(device)

if __name__ == "__main__":
    import dataset
    import torch.utils.data as data
    import time
    import os

    # Configuration for Test
    # epochs = 8 (Global is 1)
    best_model = None
    # batch_size = 32 (Global)
    TreePoint = bptt * 16

    # Dataset
    train_set = dataset.DataFolder(root=trainDataRoot, TreePoint=TreePoint, transform=None, dataLenPerFile=None)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=False, num_workers=0, drop_last=True)

    # Logger
    if not os.path.exists(checkpointPath):
        os.makedirs(checkpointPath)
    printl = CPrintl(expName + '/loss.log')
    writer = SummaryWriter('./log/' + expName)
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    printl(expComment + ' Pid: ' + str(os.getpid()))

    log_interval = 1 # concise log

    # Optimization
    criterion = nn.CrossEntropyLoss()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    best_val_loss = float("inf")
    idloss = 0

    # Reload
    saveDic = None
    if os.path.exists(checkpointPath):
         # Try to load latest?
         pass

    def train(epoch):
        print("--Starting Training--")
        global idloss, best_val_loss
        model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()

        for Batch, d in enumerate(train_loader):
            # print("--Training Batch " + str(Batch) + "--")

            # d is list of [chunk]. chunk is [TreePoint, 17, 4]?
            # dataset returns [1, TreePoint, 17, 4] if batch_size=1?
            # DataLoader collates.
            # d[0] is [Batch, TreePoint, 17, 4]

            train_data = d[0].permute(1, 0, 2, 3).to(device) # [TreePoint, Batch, 17, 4]

            src_mask = None

            for index, i in enumerate(range(0, train_data.size(0), bptt)):
                data, targets, dataFeat = get_batch(train_data, i)

                optimizer.zero_grad()
                output = model(data, src_mask, dataFeat)

                # Use Center Token Prediction
                output = output[8, :, :] # [Batch, 255]

                loss = criterion(output, targets) / math.log(2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()

                batch = index
                if batch % log_interval == 0 and batch > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    printl('| epoch {:3d} | Batch {:3d} | {:4d}/{:4d} batches | '
                           'lr {:02.2f} | ms/batch {:5.2f} | '
                           'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, Batch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                                             elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                    total_loss = 0
                    start_time = time.time()
                    writer.add_scalar('train_loss', cur_loss, idloss)
                    idloss += 1

            if Batch % 10 == 0:
                save(epoch * 100000 + Batch, saveDict={'encoder': model.state_dict(), 'idloss': idloss, 'epoch': epoch,
                                                       'best_val_loss': best_val_loss}, modelDir=checkpointPath)

    # Train Loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(epoch)
        printl('-' * 89)
        scheduler.step()
        printl('-' * 89)
