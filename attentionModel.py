
import torch
import torch.nn as nn
import math
import copy
from networkTool import device


class SelfMultiheadAttention(nn.Module):
    """Multi-head Self Attention module implementation.

    This class implements a multi-head self-attention mechanism as described in
    'Attention Is All You Need' (Vaswankar et al., 2017). It projects the input into
    query, key and value spaces, splits them into multiple heads, computes scaled
    dot-product attention, and projects back to the original space.

    Args:
        emsize (int): The embedding dimension size
        nhead (int): Number of attention heads
        dropout (float, optional): Dropout probability. Defaults to 0.5.

    Attributes:
        nhead (int): Number of attention heads
        head_size (int): Dimension of each attention head
        all_head_size (int): Total dimension of all attention heads
        mlpKey (nn.Linear): Linear transformation for key projection
        mlpQuery (nn.Linear): Linear transformation for query projection
        mlpValue (nn.Linear): Linear transformation for value projection
        dropout (nn.Dropout): Dropout layer

    Methods:
        slice(x, dim): Reshapes the input tensor for multi-head attention
        forward(em, mask): Computes self-attention over the input embeddings

    Shape:
        - Input:
            - em: :math:`(S, N, E)` where S is sequence length, N is batch size,
              and E is embedding dimension
            - mask: :math:`(S, S)` attention mask
        - Output: :math:`(S, N, E)` same shape as input embedding
    """

    def __init__(self, emsize, nhead, dropout=0.5):
        super(SelfMultiheadAttention, self).__init__()
        self.nhead = nhead  # 4
        self.head_size = emsize // nhead  # 168//4=42
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  #
        self.mlpKey = nn.Linear(emsize, self.all_head_size)  # MLP(168,168)
        self.mlpQuery = nn.Linear(emsize, self.all_head_size)
        self.mlpValue = nn.Linear(emsize, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

    # Slice the output of mlpKQV to implement multi-head attention. dim = 3(for sibling attention): input:  x.shape =
    # [batch_size, bptt, emsize]  output: x.shape = [batch_size,nhead,bptt,head_size] dim = 4(for ancestor
    # attention)：input:  x.shape = [batch_size, bptt, levelNumK , emsize]  output: x.shape = [batch_size,bptt, nhead,
    # levelNumK, head_size]
    def slice(self, x, dim):
        new_x_shape = x.size()[:-1] + (self.nhead,
                                       self.head_size)  # [batch_size, bptt, nhead, head_size] or [batch_size, bptt,
        # levelNumK, nhead, head_size]
        x = x.view(*new_x_shape)
        if dim == 3:
            x = x.permute(0, 2, 1, 3)
        elif dim == 4:
            x = x.permute(0, 1, 3, 2, 4)
        return x

    # em.shape = [bptt,batch_size,emsize]  mask.shape=[bptt, bptt]
    def forward(self, em, mask):
        em = em.transpose(0, 1).contiguous()  # [batch_size,bptt,...]
        Key = self.slice(self.mlpKey(em),
                         em.dim())  # [batch_size, bptt, all_head_size] -> [batch_size,nhead,bptt,head_size]
        Query = self.slice(self.mlpQuery(em), em.dim())  # torch.Size([32, 4, 256, 42])
        Value = self.slice(self.mlpValue(em), em.dim())

        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(
            self.head_size)  # [batch_size,nhead,bptt,bptt] or [bs,bptt,nhead,levelNumK,levelNumK]
        if mask is not None:
            attention_score = attention_score + mask

        attention_map = self.dropout(nn.Softmax(dim=-1)(attention_score))

        context = torch.matmul(attention_map,
                               Value)  # [batch_size, 4 nhead, bptt, 42 head_size] #torch.Size([32, 4, 256, 42])
        if context.dim() == 4:
            context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, bptt, 4 nhead, 42 head_size]
        elif context.dim() == 5:
            context = context.permute(0, 1, 3, 2, 4).contiguous()  # [batch_size, bptt, levelNumK, 8, 64]
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        context = context.transpose(0, 1).contiguous()
        return context


# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class TransformerLayer(nn.Module):
    """A Transformer layer implementation with self-attention and feed-forward neural network.

    This layer implements a standard Transformer layer consisting of:
    1. Multi-head self-attention mechanism
    2. Add & Norm layer
    3. Feed-forward neural network
    4. Final Add & Norm layer

    Args:
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
        dim_feedforward (int): The dimension of the feedforward network model.
        dropout (float): The dropout value.

    Attributes:
        self_attn: Self multi-head attention layer
        linear1: First linear transformation of feed-forward network
        linear2: Second linear transformation of feed-forward network
        dropout: Dropout layer for feed-forward network
        norm1: Layer normalization after attention
        norm2: Layer normalization after feed-forward
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = SelfMultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer.

        Args:
            src: input sequence.
            src_mask: attention mask.
            src_key_padding_mask: mask for padding keys.
        """
        src2 = self.self_attn(src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerModule(nn.Module):
    """
    TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of the TransformerLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerModule, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

# class SiblingAttention(nn.Module):
#     def __init__(self,  ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
#         super(SiblingAttention, self).__init__()
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(ninp, dropout)
#         encoder_layers = TransformerLayer(ninp, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerModule(encoder_layers, nlayers)
#         self.decoder = nn.Linear(ninp, ntoken)
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         #self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src, src_mask):
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.decoder(output)
#         return output
