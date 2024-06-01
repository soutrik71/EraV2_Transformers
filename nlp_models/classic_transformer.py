import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import spacy
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# Embedding Layer - Converts the input tokens to their corresponding embeddings, The dim of the embedding is d_model which should be 1 more than the vocab size


class IOEmbedding(nn.Module):
    """
    This class is used to create the embedding layer for the input and output tokens.
    The embedding layer is a simple nn.Embedding layer with the input size as the vocab size and the output size as the d_model.
    The output of the embedding layer is then multiplied by sqrt(d_model) as per the transformer paper.

    Args:
    d_model : int : The dimension of the model
    vocab_size : int : The size of the vocabulary

    Returns:
    op : tensor : The output of the embedding layer
    """

    def __init__(self, d_model, vocab_size):
        super(IOEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        op = self.embedding_layer(x) * math.sqrt(self.d_model)
        return op


# Positional Encoding Layer - The positional encoding layer is used to add the positional information to the input sequence. The positional encoding is added to the input sequence by simply adding the positional encoding to the input sequence. The positional encoding is a tensor of shape (1, seq_len, d_model) which is added to the input sequence.


def create_positional_embeddings(seq_len, d_model):
    """
    This function is used to create the positional embeddings for the input sequence.
    The positional embeddings are created using the formulae mentioned in the transformer paper.
    The positional embeddings are then returned.

    Args:
    seq_len : int : The length of the sequence
    d_model : int : The dimension of the model

    Returns:
    positional_encoding : tensor : The positional embeddings for the input sequence
    """
    pos = torch.arange(0, seq_len).unsqueeze(1)
    two_index = torch.arange(0, d_model, 2).float()
    exponential_term = torch.exp(two_index * -(math.log(10000.0) / d_model))
    positional_encoding = torch.zeros(seq_len, d_model)
    positional_encoding[:, 0::2] = torch.sin(pos.float() * exponential_term)
    positional_encoding[:, 1::2] = torch.cos(pos.float() * exponential_term)
    return positional_encoding


class PositionalEncodingBlock(nn.Module):
    """
    This class is used to add the positional encoding to the input sequence.
    The positional encoding is added to the input sequence by simply adding the positional encoding to the input sequence.
    The positional encoding is a tensor of shape (1, seq_len, d_model) which is added to the input sequence.

    Args:
    d_model : int : The dimension of the model
    seq_len : int : The length of the sequence

    Returns:
    x : tensor : The output of the positional encoding block
    """

    def __init__(self, d_model, seq_len):
        super(PositionalEncodingBlock, self).__init__()
        positional_encoding = create_positional_embeddings(seq_len, d_model)
        # add batch information by introducing a new dimension
        positional_encoding = positional_encoding.unsqueeze(0)
        # Use register buffer to save positional_encoding when the model is exported in disk.
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        # add positional encoding to the input which is ouput of the embedding layer
        x = x + self.positional_encoding[:, : x.size(1), :].requires_grad_(False)
        return x


# Masked Multi-Head Attention Block - The masked multi-head attention block is used to calculate the attention scores for the input query, key and value. The attention scores are then returned. The masked multi-head attention block consists of the following layers: - Wq : The query weights - Wk : The key weights - Wv : The value weights - Wo : The output weights - Dropout : The dropout layer The forward function of the class is used to calculate the attention scores for the input query, key and value. The attention scores are then returned.


class MaskedMultiHeadAttentionBlock(nn.Module):
    """
    This class is used to create the masked multi-head attention block.
    The masked multi-head attention block consists of the following layers:
    - Wq : The query weights
    - Wk : The key weights
    - Wv : The value weights
    - Wo : The output weights
    - Dropout : The dropout layer
    The forward function of the class is used to calculate the attention scores for the input query, key and value.
    The attention scores are then returned.

    Args:
    d_model : int : The dimension of the model
    heads : int : The number of heads
    dropout_rate : float : The dropout rate

    Returns:
    op : tensor : The output of the masked multi-head attention block
    """

    def __init__(self, d_model: int, heads: int, dropout_rate: float) -> None:
        super(MaskedMultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.heads = heads
        # init head dim
        self.d_k = d_model // heads

        # initialize fc layers Weights for q,k,v generation
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        # Final Dense Layer for Condensation after application of attention
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def multihead_attention(self, q, k, v, mask, dropout):
        # multiply query and key vector (batch, head, seq , d_k) @ (batch , head, d_k, seq) = (batch, head, seq, seq)
        d_k = q.shape[-1]
        qk = q @ k.transpose(-2, -1)
        scaled_qk = qk / math.sqrt(d_k)

        # application of masking
        if mask is not None:
            print("Applying Mask to the scaled_qk")
            scaled_qk.masked_fill_(mask == 0, -1e9)

        # softmax scoring with upper triangular masking if applicable
        scores = F.softmax(scaled_qk, dim=-1)
        # print(scores)

        if dropout is not None:
            print("Applying Dropout to the softmax scores")
            scores = dropout(scores)
            # print(scores)

        # (batch, head, seq_len, seq_len) @ # (batch, head, seq_len, d_model) = (batch, head, seq_len, d_model)
        final_matrix = scores @ v
        # print(final_matrix)

        return final_matrix

    def forward(self, q, k, v, mask=None):
        # fisrt we calculate the query, key and value vectors from fc layers then convert them into groups and then calculate the attention scores and concat back
        query = self.Wq(q)
        key = self.Wk(k)
        value = self.Wv(v)

        # convert (batch, seq, d_model) -> (batch, seq, head, d_k) -> (batch, head, seq, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.heads, self.d_k
        ).transpose(1, 2)

        attention_scores = self.multihead_attention(
            query, key, value, mask, self.dropout
        )

        # convert (batch, head, seq, d_k) -> (batch, seq, head, d_k) -> (batch, seq, d_model)

        x = (
            attention_scores.transpose(1, 2)
            .contiguous()
            .view(attention_scores.shape[0], -1, self.d_k * self.heads)
        )
        # print(x.shape)

        op = self.Wo(x)

        return op


# Feed Forward Block - The feed forward block is used to calculate the output of the feed forward network. The feed forward block consists of the following layers: - Linear1 : The first linear layer - Dropout : The dropout layer - Linear2 : The second linear layer The forward function of the class is used to calculate the output of the feed forward block. The output of the feed forward block is then returned.


class FeedForwardBlock(nn.Module):
    """
    This class is used to create the feed forward block.
    The feed forward block consists of the following layers:
    - Linear1 : The first linear layer
    - Dropout : The dropout layer
    - Linear2 : The second linear layer
    The forward function of the class is used to calculate the output of the feed forward block.
    The output of the feed forward block is then returned.

    Args:
    d_model : int : The dimension of the model
    d_ff : int : The dimension of the feed forward network
    dropout_rate : float : The dropout rate

    Returns:
    out : tensor : The output of the feed forward block
    """

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super(FeedForwardBlock, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)

        return out


# Add Norm Block - The add norm block is used to add the input to the output of the sublayer and then apply layer normalization


# Layer Normalization Block - The layer normalization block is used to normalize the input tensor. The layer normalization block consists of the following layers: - LayerNorm : The layer normalization layer The forward function of the class is used to calculate the output of the layer normalization block. The output of the layer normalization block is then returned.


class LayerNormalizationBlock(nn.Module):
    """
    This class is used to create the layer normalization block.
    The layer normalization block consists of the following layers:
    - LayerNorm : The layer normalization layer
    The forward function of the class is used to calculate the output of the layer normalization block.
    The output of the layer normalization block is then returned.

    Args:
    features : int : The number of features
    eps : float : The epsilon value

    Returns:
    x : tensor : The output of the layer normalization block
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNormalizationBlock, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # the summary metrics like mean and variance are calculated across the last dimension ie embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # normalization
        norm = (x - mean) / (std + self.eps)
        # scale and shift
        op = self.alpha * norm + self.bias
        return op


# Skip Connection Block - The skip connection block is used to create the skip connection between the input and the output of the sublayer. The skip connection block consists of the following layers: - Dropout : The dropout layer - LayerNorm : The layer normalization layer The forward function of the class is used to calculate the output of the skip connection block. The output of the skip connection block is then returned.


class SkipConnectionBlock(nn.Module):
    """
    This class is used to create the skip connection block.
    The skip connection block consists of the following layers:
    - Dropout : The dropout layer
    - LayerNorm : The layer normalization layer
    The forward function of the class is used to calculate the output of the skip connection block.
    The output of the skip connection block is then returned.

    Args:
    features : int : The number of features
    dropout_rate : float : The dropout rate

    Returns:
    op : tensor : The output of the skip connection block
    """

    def __init__(self, features: int, dropout_rate: float):
        super(SkipConnectionBlock, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = LayerNormalizationBlock(features)

    def forward(self, x, sublayer):
        # sublayer can be  FeedForwardBlock or MaskedMultiHeadSelfAttentionBlock
        op = x + self.dropout(sublayer(self.layernorm(x)))
        return op


# Encoder and Decoder Block


def clones(module, N):
    "Produce N identical layers of pytorch modules as a list"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# Encoder Block - The encoder block is used to create the encoder block. The encoder block consists of the following layers: - attention_block : The masked multi-head attention block - feed_forward_block : The feed forward block - skipconnection_block : The skip connection block The forward function of the class is used to calculate the output of the encoder block. The output of the encoder block is then returned.


class EncoderBlock(nn.Module):
    """
    This class is used to create the encoder block.
    The encoder block consists of the following layers:
    - attention_block : The masked multi-head attention block
    - feed_forward_block : The feed forward block
    - skipconnection_block : The skip connection block
    The forward function of the class is used to calculate the output of the encoder block.
    The output of the encoder block is then returned.

    Args:
    features : int : The number of features
    attention_block : nn.Module : The masked multi-head attention block
    feed_forward_block : nn.Module : The feed forward block
    dropout_rate : float : The dropout rate

    Returns:
    x : tensor : The output of the encoder block
    """

    def __init__(
        self,
        features: int,
        attention_block: MaskedMultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout_rate: float,
    ):
        super(EncoderBlock, self).__init__()
        self.features = features
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.skipconnection_block = clones(
            SkipConnectionBlock(features, dropout_rate), 2
        )  # always 2 skip connections

    def forward(self, x, mask):
        x = self.skipconnection_block[0](
            x, lambda x: self.attention_block(k=x, q=x, v=x, mask=mask)
        )
        x = self.skipconnection_block[1](x, self.feed_forward_block)
        return x


# Encoder Sequence Block - The encoder sequence block is used to create the encoder sequence block. The encoder sequence block consists of the following layers: - encoder_blocks : The encoder blocks - layernorm : The layer normalization block The forward function of the class is used to calculate the output of the encoder sequence block. The output of the encoder sequence block is then returned.


class EncoderSequenceBlock(nn.Module):
    """
    This class is used to create the encoder sequence block.
    The encoder sequence block consists of the following layers:
    - encoder_blocks : The encoder blocks
    - layernorm : The layer normalization block
    The forward function of the class is used to calculate the output of the encoder sequence block.

    Args:
    encoder_block : nn.Module : The encoder block

    Returns:
    x : tensor : The output of the encoder sequence block
    """

    def __init__(self, encoder_block: EncoderBlock, N: int):
        super(EncoderSequenceBlock, self).__init__()
        self.encoder_blocks = clones(encoder_block, N)
        self.layernorm = LayerNormalizationBlock(encoder_block.features)

    def forward(self, x, mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)

        return self.layernorm(x)


# Decoder Block - The decoder block is used to create the decoder block. The decoder block consists of the following layers: - attention_block : The masked multi-head attention block - cross_attention_block : The masked multi-head attention block - feed_forward_block : The feed forward block - skipconnection_block : The skip connection block The forward function of the class is used to calculate the output of the decoder block. The output of the decoder block is then returned.


class DecoderBlock(nn.Module):
    """
    This class is used to create the decoder block.
    The decoder block consists of the following layers:
    - attention_block : The masked multi-head attention block
    - cross_attention_block : The masked multi-head attention block
    - feed_forward_block : The feed forward block
    - skipconnection_block : The skip connection block
    The forward function of the class is used to calculate the output of the decoder block.
    The output of the decoder block is then returned.

    Args:
    features : int : The number of features
    attention_block : nn.Module : The masked multi-head attention block
    cross_attention_block : nn.Module : The masked multi-head attention block
    feed_forward_block : nn.Module : The feed forward block
    dropout_rate : float : The dropout rate

    Returns:
    x : tensor : The output of the decoder block
    """

    def __init__(
        self,
        features: int,
        attention_block: MaskedMultiHeadAttentionBlock,
        cross_attention_block: MaskedMultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout_rate: float,
    ):
        super(DecoderBlock, self).__init__()
        self.features = features
        self.attention_block = attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.skip_connection_block = clones(
            SkipConnectionBlock(features, dropout_rate), 3
        )

    def forward(self, x, last_encoder_output, src_mask, tgt_mask):
        # atttention block takes the last decoder op
        x = self.skip_connection_block[0](
            x, lambda x: self.attention_block(k=x, q=x, v=x, mask=tgt_mask)
        )
        # cross attention block takes the op from entire encoder layer
        x = self.skip_connection_block[1](
            x,
            lambda x: self.cross_attention_block(
                q=x, k=last_encoder_output, v=last_encoder_output, mask=src_mask
            ),
        )
        # feed forward block
        x = self.skip_connection_block[2](x, self.feed_forward_block)
        return x


# Decoder Sequence Block - The decoder sequence block is used to create the decoder sequence block. The decoder sequence block consists of the following layers: - decoder_blocks : The decoder blocks - layernorm : The layer normalization block The forward function of the class is used to calculate the output of the decoder sequence block. The output of the decoder sequence block is then returned.


class DecoderSequenceBlock(nn.Module):
    """
    This class is used to create the decoder sequence block.
    The decoder sequence block consists of the following layers:
    - decoder_blocks : The decoder blocks
    - layernorm : The layer normalization block
    The forward function of the class is used to calculate the output of the decoder sequence block.

    Args:
    decoder_block : nn.Module : The decoder block

    Returns:
    x : tensor : The output of the decoder sequence block
    """

    def __init__(self, decoder_block: DecoderBlock, N: int):
        super(DecoderSequenceBlock, self).__init__()
        self.decoder_blocks = clones(decoder_block, N)
        self.layernorm = LayerNormalizationBlock(decoder_block.features)

    def forward(self, x, last_encoder_output, src_mask, tgt_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, last_encoder_output, src_mask, tgt_mask)

        return self.layernorm(x)


# Linear Softmax Block - The linear softmax block is used to create the linear softmax block. The linear softmax block consists of the following layers: - linear : The linear layer The forward function of the class is used to calculate the output of the linear softmax block. The output of the linear softmax block is then returned.


class LinearSoftmaxBlock(nn.Module):
    """
    This class is used to create the linear softmax block.
    The linear softmax block consists of the following layers:
    - linear : The linear layer
    The forward function of the class is used to calculate the output of the linear softmax block.
    The output of the linear softmax block is then returned.

    Args:
    d_model : int : The dimension of the model
    vocab_size : int : The size of the vocabulary

    Returns:
    op : tensor : The output of the linear softmax block
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Softmax using the last dimension
        # [batch, seq_len, d_model] -> [batch, seq_len]
        return torch.log_softmax(self.linear(x), dim=-1)


# Transformer Model - The transformer model is used to create the transformer model. The transformer model consists of the following layers: - input_embedding_block : The input embedding block - output_embedding_block : The output embedding block - input_pos_block : The positional encoding block for the input sequence - output_pos_block : The positional encoding block for the output sequence - encoder_seq_block : The encoder sequence block - decoder_seq_block : The decoder sequence block - linear_softmax_block : The linear softmax block The forward function of the class is used to calculate the output of the transformer model.


class Transformer(nn.Module):
    """
    This class is used to create the transformer model.
    The transformer model consists of the following layers:
    - input_embedding_block : The input embedding block
    - output_embedding_block : The output embedding block
    - input_pos_block : The positional encoding block for the input sequence
    - output_pos_block : The positional encoding block for the output sequence
    - encoder_seq_block : The encoder sequence block
    - decoder_seq_block : The decoder sequence block
    - linear_softmax_block : The linear softmax block
    The forward function of the class is used to calculate the output of the transformer model.

    Args:
    input_embedding_block : nn.Module : The input embedding block
    output_embedding_block : nn.Module : The output embedding block
    input_pos_block : nn.Module : The positional encoding block for the input sequence
    output_pos_block : nn.Module : The positional encoding block for the output sequence
    encoder_seq_block : nn.Module : The encoder sequence block
    decoder_seq_block : nn.Module : The decoder sequence block
    linear_sotfmax_block : nn.Module : The linear softmax block

    Returns:
    op : tensor : The output of the transformer model
    """

    def __init__(
        self,
        input_embedding_block: IOEmbedding,
        output_embedding_block: IOEmbedding,
        input_pos_block: PositionalEncodingBlock,
        output_pos_block: PositionalEncodingBlock,
        encoder_seq_block: EncoderSequenceBlock,
        decoder_seq_block: DecoderSequenceBlock,
        linear_sotfmax_block: LinearSoftmaxBlock,
    ):
        super(Transformer, self).__init__()
        self.input_embedding_block = input_embedding_block
        self.output_embedding_block = output_embedding_block
        self.input_pos_block = input_pos_block
        self.output_pos_block = output_pos_block
        self.encoder_seq_block = encoder_seq_block
        self.decoder_seq_block = decoder_seq_block
        self.linear_sotfmax_block = linear_sotfmax_block

    def encode(self, src, src_mask):
        src = self.input_embedding_block(src)
        src = self.input_pos_block(src)
        return self.encoder_seq_block(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.output_embedding_block(tgt)
        tgt = self.output_pos_block(tgt)
        return self.decoder_seq_block(tgt, encoder_output, src_mask, tgt_mask)

    def linear_projection(self, x):
        return self.linear_sotfmax_block(x)


# Helper Function - The helper function is used to build the transformer model. The transformer model consists of the following layers: - masked_multi_head_attention_block : The masked multi-head attention block - feed_forward_block : The feed forward block - input_embedding_block : The input embedding block - output_embedding_block : The output embedding block - input_pos_block : The positional encoding block for the input sequence - output_pos_block : The positional encoding block for the output sequence - encoder_block : The encoder block - encoder_seq_block : The encoder sequence block - decoder_block : The decoder block - decoder_seq_block : The decoder sequence block - linear_block : The linear softmax block The transformer model is then returned.


def build_transformer(d_model, heads, N, d_ff, dropout_rate, dim):
    """
    This function is used to build the transformer model.
    The transformer model consists of the following layers:
    - masked_multi_head_attention_block : The masked multi-head attention block
    - feed_forward_block : The feed forward block
    - input_embedding_block : The input embedding block
    - output_embedding_block : The output embedding block
    - input_pos_block : The positional encoding block for the input sequence
    - output_pos_block : The positional encoding block for the output sequence
    - encoder_block : The encoder block
    - encoder_seq_block : The encoder sequence block
    - decoder_block : The decoder block
    - decoder_seq_block : The decoder sequence block
    - linear_block : The linear softmax block
    The transformer model is then returned.

    Args:
    d_model : int : The dimension of the model
    heads : int : The number of heads
    N : int : The number of encoder and decoder blocks
    d_ff : int : The dimension of the feed forward network
    dropout_rate : float : The dropout rate
    dim : int : The size of the vocabulary

    Returns:
    model : nn.Module : The transformer model
    """
    c = copy.deepcopy
    # masked multi-head attention block
    attention_block = MaskedMultiHeadAttentionBlock(
        d_model=d_model, heads=heads, dropout_rate=dropout_rate
    )
    print(attention_block)
    # feed forward block
    feed_forward_block = FeedForwardBlock(
        d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate
    )
    print(feed_forward_block)
    # input and output embedding block
    input_embedding_block = IOEmbedding(d_model=d_model, vocab_size=dim)
    output_embedding_block = IOEmbedding(d_model=d_model, vocab_size=dim)
    # positional encoding block
    input_pos_block = PositionalEncodingBlock(d_model=d_model, seq_len=dim)
    output_pos_block = PositionalEncodingBlock(d_model=d_model, seq_len=dim)

    # encoder block
    encoder_block = EncoderBlock(
        d_model, c(attention_block), c(feed_forward_block), dropout_rate
    )
    print(encoder_block)
    # encoder seq block
    encoder_seq_block = EncoderSequenceBlock(encoder_block, N)
    print(encoder_seq_block)

    # decoder block
    decoder_block = DecoderBlock(
        d_model,
        c(attention_block),
        c(attention_block),
        c(feed_forward_block),
        dropout_rate,
    )
    print(decoder_block)

    # decoder seq block
    decoder_seq_block = DecoderSequenceBlock(decoder_block, N)
    print(decoder_seq_block)

    # linear softmax block
    linear_block = LinearSoftmaxBlock(d_model=d_model, vocab_size=dim)

    # transformer model
    model = Transformer(
        input_embedding_block,
        output_embedding_block,
        input_pos_block,
        output_pos_block,
        encoder_seq_block,
        decoder_seq_block,
        linear_block,
    )

    return model
