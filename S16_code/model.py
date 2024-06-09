import torch
import torch.nn as nn
import torch.functional as F
import math
import gc
import os
from S16_code.config import get_config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:12240"
torch.backends.cudnn.benchmark = True


torch.cuda.empty_cache()
gc.collect()
torch.cuda.synchronize()


# Input Embeddings- This class is responsible for creating the input embeddings for the model
class InputEmbeddings(nn.Module):
    """
    This class is responsible for creating the input embeddings for the model
    """

    def __init__(self, d_model: int, vocab_size: int):
        super(InputEmbeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        op = self.embedding(x) * math.sqrt(self.d_model)
        return op


# Positional Encoding- This class is responsible for creating the positional encoding for the model
class PositionalEncoding(nn.Module):
    """This class is responsible for creating the positional encoding for the model"""

    def __init__(self, d_model: int, seq_len: int, dropout_rate: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(
            dropout_rate
        )  # added post we add embeddings to the positional encoding

        pe = self._postional_encoding_matrix()

        self.register_buffer("pe", pe)

    def _postional_encoding_matrix(self):
        # we will multiply a position matrix pos with dimension (seq_len,1) with an encoding matrix with dimension (1, d_model) to get a matrix with dimension (seq_len, d_model)
        position = torch.arange(0, self.seq_len).unsqueeze(1)  # (seq_len, 1)

        # to create the embedding matrix we will use d_model/2 as the dimension of the sin and cos functions
        embedding = torch.arange(0, self.d_model, 2).float()  # (d_model/2)

        # next we will create the devision factor for the positional encoding
        div_term = torch.exp(embedding * -(math.log(10000.0) / self.d_model))

        # create a postional encoding matrix of dimension (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)

        # apply sin to even indices and cos to odd indices of the position matrix

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # in order to add the positional encoding to the input embeddings we need to add a batch dimension to the positional encoding matrix
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        return pe

    def forward(self, x):
        op = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(op)


# MultiHeadAttentionBlock- This class is responsible for creating the multi head attention block
class MultiHeadAttentionBlock(nn.Module):
    """This class is responsible for creating the multi head attention block"""

    def __init__(self, d_model: int, h: int, dropout_rate: float):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout_rate)

        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = (
            d_model // h
        )  # dimension for query, key and value matrix under each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention_block(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        # each matrix after head split is of size (batch, h, seq_len, d_k), so (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) = (batch, h, seq_len, seq_len)
        qk = query @ key.transpose(-2, -1)
        scaled_qk = qk / math.sqrt(d_k)

        if mask is not None:
            # where mask is 0, we will replace the value with -1e9
            scaled_qk.masked_fill_(mask == 0, -1e9)

        attention = torch.softmax(scaled_qk, dim=-1)

        if dropout is not None:
            attention = dropout(attention)

        # the output should be same as input value matrix, so (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) = (batch, h, seq_len, d_k)

        scores = attention @ value

        return scores, attention

    def forward(self, q, k, v, mask):
        # initally q,k,v are in the shape of (batch, seq_len, d_model) , we need to convert them into the shape of (batch, h, seq_len, d_k)
        # before that we have to add weights to q,k,v using the linear layers
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # the conversion is (batch,seq_len,d_model) -> (batch, seq_len , h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        # print(query.shape)
        # print(key.shape)
        # print(value.shape)

        # next we calculate attention score
        x, self.attention_scores = MultiHeadAttentionBlock.attention_block(
            query, key, value, mask, self.dropout
        )

        # x is in the shape of (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        op = self.w_o(x)
        return op


# LayerNormalization- This class is responsible for creating the layer normalization block
class LayerNormalization(nn.Module):
    """This class is responsible for creating the layer normalization block"""

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.features = features
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(self.features))
        self.beta = nn.Parameter(torch.zeros(self.features))

    def forward(self, x):
        # print(f"The shape of x in layer Norm is {x.shape}")
        # x is of shape (batch, seq_len, hidden_dim)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        norm_op = self.alpha * (x - mean) / (std + self.eps) + self.beta
        return norm_op


# ResidualConnection- This class is responsible for creating the residual connection block ie Pre-Layer Normalization block
class ResidualConnection(nn.Module):
    """This class is responsible for creating the residual connection block ie Pre-Layer Normalization block"""

    def __init__(self, features: int, dropout_rate: float):
        super(ResidualConnection, self).__init__()
        self.features = features
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = LayerNormalization(self.features)

    def forward(self, x, sublayer):
        residual_op = x + self.dropout(sublayer(self.norm(x)))
        return residual_op


# FeedForwardBlock- This class is responsible for creating the feed forward block
class FeedForwardBlock(nn.Module):
    """This class is responsible for creating the feed forward block"""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# EncoderBlock- This class is responsible for creating the encoder block
class EncoderBlock(nn.Module):
    """This class is responsible for creating the encoder block"""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout_rate: float,
    ):
        super(EncoderBlock, self).__init__()
        self.features = features
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(self.features, dropout_rate) for _ in range(2)]
        )  # 2 residual connections in the encoder block

    def forward(self, x, src_mask):
        # print(f"The shape of x in encoder block is {x.shape}")
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )  # apply self attention block on the input x as query, key and value

        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


# Encoder- This class is responsible for creating the encoder sequence of the transformer model
class Encoder(nn.Module):
    """This class is responsible for creating the encoder sequence of the transformer model"""

    def __init__(self, features: int, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        # print(f"The shape of x in encoder-seq is {x.shape}")
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# DecoderBlock- This class is responsible for creating the decoder block of the transformer model
class DecoderBlock(nn.Module):
    """This class is responsible for creating the decoder block of the transformer model"""

    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout_rate: float,
    ):
        super(DecoderBlock, self).__init__()
        self.features = features
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(self.features, dropout_rate) for _ in range(3)]
        )  # 3 residual connections in the decoder block

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # print(f"The shape of x in decoder block is {x.shape}")
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )  # self attention block with x as query, key and value
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )  # cross attention block with encoder output as key and value and x as query
        x = self.residual_connections[2](
            x, self.feed_forward_block
        )  # feed forward block

        return x


# Decoder- This class is responsible for creating the decoder sequence of the transformer model
class Decoder(nn.Module):
    """This class is responsible for creating the decoder sequence of the transformer model"""

    def __init__(self, features: int, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # print(f"The shape of x in decoder-seq is {x.shape}")
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


# ProjectionLayer- This class is responsible for creating the projection layer of the transformer model
class ProjectionLayer(nn.Module):
    """This class is responsible for creating the projection layer of the transformer model"""

    def __init__(
        self, d_model: int, vocab_size: int
    ) -> None:  # Model dimension and the size of the output vocabulary
        super().__init__()
        self.proj = nn.Linear(
            d_model, vocab_size
        )  # Linear layer for projecting the feature space of 'd_model' to the output space of 'vocab_size'

    def forward(self, x):
        return torch.log_softmax(
            self.proj(x), dim=-1
        )  # Applying the log Softmax function to the output


# TransformerModel- This class is responsible for creating the transformer model
class TransformerModel(nn.Module):
    """This class is responsible for creating the transformer model"""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_position: PositionalEncoding,
        tgt_position: PositionalEncoding,
        generator: ProjectionLayer,
    ):
        super(TransformerModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.generator = generator

    def encode(self, src, src_mask):
        # print(f"The shape of src in encode is {src.shape}")
        x = self.src_embed(src)  # apply the input embeddings to input source lng
        # print(f"The shape after src_embed in encode is {x.shape}")
        x = self.src_position(
            x
        )  # apply the positional encoding to the input source lng
        # print(f"The shape after src_position in encode is {x.shape}")
        op = self.encoder(
            x, src_mask
        )  # apply the encoder block to the input source lng

        return op

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # print(f"The shape of tgt in decode is {tgt.shape}")
        tgt = self.tgt_embed(tgt)  # apply the input embeddings to input target lng

        # print(f"The shape after tgt_embed in decode is {tgt.shape}")
        tgt = self.tgt_position(
            tgt
        )  # apply the positional encoding to the input target lng
        # print(f"The shape after tgt_position in decode is {tgt.shape}")

        # Returning the target embeddings, the output of the encoder, and both source and target masks The target mask ensures that the model won't 'see' future elements of the sequence
        op = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )  # query = tgt, key = encoder_output, value = encoder_output
        return op

    def project(self, x):
        return self.generator(x)


# build_transformer- This function is responsible for building the transformer model
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int = get_config()["seq_len"],
    tgt_seq_len: int = get_config()["seq_len"],
    d_model: int = get_config()["d_model"],
    N: int = get_config()["N"],
    h: int = get_config()["h"],
    dropout_rate: float = get_config()["dropout_rate"],
    d_ff: int = get_config()["d_ff"],
):
    """
    This function is responsible for building the transformer model
    """
    # first we create the embedding layer for both encoder and decoder
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # next we create the positional encoding layer for both encoder and decoder
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout_rate)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout_rate)

    # create the encoder block for N times and feed it to the encoder sequence using nn.ModuleList
    encoder_layers = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_rate)
        encoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        encoder_block = EncoderBlock(
            d_model,
            encoder_self_attention_block,
            encoder_feed_forward_block,
            dropout_rate,
        )
        encoder_layers.append(encoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_layers))

    # create the decoder block for N times and feed it to the decoder sequence using nn.ModuleList

    decoder_layers = []

    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_rate)
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout_rate
        )
        decoder_feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_rate)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            decoder_feed_forward_block,
            dropout_rate,
        )
        decoder_layers.append(decoder_block)

    decoder = Decoder(d_model, nn.ModuleList(decoder_layers))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create the final transformer model
    transformer_model = TransformerModel(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_model
