import os
import itertools
import random
import logging
from typing import Generator as TGenerator
from typing import Tuple, List

from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
from spacy import Language
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchinfo


# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")


logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING").upper())
logger = logging.getLogger()


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers followed by a LayerNorm (subtracts the
    mean and divides by stdev times eps before running through a simple linear
    layer; see citations/source)"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Normalizes output, runs it through a dropout layer, and adds input. For
    code simplicity the norm is first as opposed to last. Eq: x + dropout(sublayer(norm(x))
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        global logger
        logger.info(f"EncoderLayer forward:  x dim {x.size()}  mask dim {mask.size()}")
        # exit(1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        logger.info(f"encoder x dim: {x.size()}")

        x_final = self.sublayer[1](x, self.feed_forward)
        logger.info(f"encoder x_final dim: {x_final.size()}")

        return x_final


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # There are 3 fully connected sublayers in the decoder: after
        # self-attention, after the src (encoder) attention, and finally after
        # decoder output.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, target, memory, src_mask, tgt_mask):
        """The decoder always takes as input output target and memory from the
        encoder's final output. It then runs self-attention before running
        self+src attention."""
        m = memory
        x = self.sublayer[0](target, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other
    models.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Process masked src and target sequences. The input is fully encoded
        before being processed by the decoder.

        @src: b x max_padding  (LongTensor)
        @tgt: b x max_padding  (LongTensor)
        @src_mask: b x 1 x (max_padding-1)  (BoolTensor)
        @tgt_mask: b x (max_padding-1) x 1  (BoolTensor)

        The input @src and @tgt are both simply integer lookups; the Embedding layer is used
        to lookup each of these vectors from the model.
        I don't know why the masks are max_padding-1, but presumably because there is always at least one word.

        Per masking, there are basically three cases which should be detailed
        separately where used in the code:
            1) causal masking to ensure the decoder learns to function
               autoregressively by depriving it of information on future output
               tokens
            2) padding token masking to ensure that padding tokens are not
               learnt'
            3) other masking to perform custom training to attend to specific
               tokens for domain oriented tasks
        """
        global logger
        logger.info(f"encdec src: {src.size()}  {src.type()}")
        logger.info(f"encdec tgt: {tgt.size()}  {tgt.type()}")
        logger.info(f"encdec src_mask: {src_mask.size()}  {src_mask.type()}")
        logger.info(f"encdec tgt_mask: {tgt_mask.size()}  {tgt_mask.type()}")

        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """encode runs the encoder layers and gets their final output.

        @src:
        @src_mask:
        """
        global logger
        logger.info(
            f"encdec encoder layer: src.size()={src.size()} src_mask.size()={src_mask.size()}"
        )
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """decode runs the entire decoder layers, taking the encoder's final output as its initial input.

        @memory: size is (b x max_padding x d_model)
        @src_mask: size is (b x 1 x d_model)
        @tgt: size varies according to how much previous context (previous words, 'c') is loaded, (1 x c)
        @tgt_mask: size is coupled to tgt size, and is a triangular matrix whose elements above the diagonal are true.
          The size per @tgt is (1 x c x c)
        """
        global logger
        logger.info(
            f"encdec decoder layer: memory.size()={memory.size()} src_mask.size()={src_mask.size()} tgt.size()={tgt.size()} tgt_mask.size()={tgt_mask.size()}"
        )

        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def subsequent_mask(size):
    """Returns a tensor of size (1 x @size x @size). Mask out subsequent positions.
    Output embeddings are offset by one position. This returns an upper
    triangular matrix of booleans whose diagonal and below-diagonal entries are
    False."""

    # Note: the '1' first dim is inferred by torch to be for batch, i.e.
    # torch.ones((1,2,2)) is '[[[1, 1],[0, 1]]]'
    attn_shape = (1, size, size)
    # Creates an upper triangular matrix whose diagonal is zero; the 'diagonal'
    # param indicates how many diagonals above the main diagonal to set to zero.
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # Converts the forementioned matrix' 1s entries to True, 0s to False.
    return subsequent_mask == 0


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


def attention(query, key, value, mask=None, dropout=None):
    """
    Per the masking issue: it might be worth looking at https://github.com/harvardnlp/annotated-transformer/issues/137.
    There is a reported issue with make_std_mask. The first-principles definition
    of std-mask has not been reviewed / is the most opaque part of this code.

    An attention function maps a query and a set of key-value pairs to an
    output, where the query, keys, values, and output are all vectors.  The
    output is computed as a weighted sum of the values, where the weight
    assigned to each value is computed by a compatibility function of the query
    with the corresponding key.

    We call our particular attention "Scaled Dot-Product Attention". The input
    consists of queries and keys of dimension $d_k$, and values of dimension
    $d_v$.  We compute the dot products of the query with all keys, divide each
    by $sqrt{d_k}$, and apply a softmax function to obtain the weights on the
    values.

    Args:
        * query: 1 x h x seq_len x b
        * key:   1 x h x seq_len x b
        * value: 1 x h x seq_len x b
        * mask:
        * dropout: if any
    """
    global logger
    # Get the dimensionality d_head
    d_k = query.size(-1)
    k_t = key.transpose(-2, -1)

    logger.info(
        f"In attn: d_k={d_k} q={query.size()} k={key.size()} k_t=k.transpose(-2,-1)={k_t.size()}"
    )

    scores = torch.matmul(query, k_t) / math.sqrt(d_k)

    """
    Output during training:
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 71, 72])
        mask.size=torch.Size([8, 1, 71, 71]) scores=torch.Size([8, 8, 71, 71])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 71, 72])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 72, 72])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 72, 72])
        mask.size=torch.Size([8, 1, 71, 71]) scores=torch.Size([8, 8, 71, 71])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 71, 72])
        mask.size=torch.Size([8, 1, 71, 71]) scores=torch.Size([8, 8, 71, 71])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 71, 72])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 72, 72])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 72, 72])
        mask.size=torch.Size([8, 1, 71, 71]) scores=torch.Size([8, 8, 71, 71])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 71, 72])

    Output time output:
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 72, 72])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([8, 8, 72, 72])
        mask.size=torch.Size([1, 1, 1, 1]) scores=torch.Size([1, 8, 1, 1])
        mask.size=torch.Size([8, 1, 1, 72]) scores=torch.Size([1, 8, 1, 576])
    """

    if mask is not None:
        # mask.size=torch.Size([32, 1, 1, 72]) scores=torch.Size([1, 8, 1,
        # 2304]) RuntimeError: The size of tensor a (72) must match the size of
        # tensor b (2304) at non-singleton dimension 3 Note that 72 * 32 = 2304
        # logger.info(f"mask.size={mask.size()} scores={scores.size()}")
        #
        # Set masked scores to negative large-numbers, such that their output
        # probs are effectively zero.

        old_shape = mask.shape
        if mask.size()[-1] != scores.size()[-1]:
            mask = mask.view(1, 1, 1, -1)
            logger.warning(
                f"mask size {old_shape} != scores size {scores.size()}, reshaped to {mask.size()}. This is currently needed during inference because ys has no batch dim."
            )

        logger.warning(
            f"q={query.size()} k={key.size()} k_t={k_t.size()} v={value.size()}"
            + f"  scores={scores.size()} mask={mask.size()} mask_old_shape={old_shape}"
        )

        # Bug cope: force the mask to satisfy size reqs, without knowing what I'm doing a priori...
        # if mask.shape[-1] == scores.shape[-1]:
        #     mask = mask.view(1, 1, 1, -1)

        # TODO: this is bug cope due to a bug with mismatched dimensions at some step in the model.
        # if mask.shape[-1] == scores.shape[-1]:
        scores = scores.masked_fill(mask == False, -1e9)
        # pass
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multihead attention allows the model to jointly attend to information from
    different representation subspaces at different positions. With a single
    attention head, averaging inhibits this. The linear weights on each Q, K, V
    weight matrix are size d_model, but are simply transformed to the space of
    the heads when needed.

        Multihead(Q,K,V) = Concat(head_1, head_2, ... head_h) * W^0 where
        head_i = Attention(QW_iq, KW_ik, VW_iv) where W_iq in R[d_model x d_k],
        W_ik in R[d_model x d_k], and W_iv in R[h*d_v x d_model] Basically, the
        matrices are the sizes of the original matrices scaled down by h, and
        such that the row/col math works out the same.
    """

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert (
            d_model % h == 0
        ), "d_model % h must equal 0; h must divide d_model but got: h={h} d_model={d_model}"
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # Create 3 linear layers for each of Q, K, and V matrices, plus one for
        # the final output, 4 total. These are the weights by which the linear
        # projection is performed for multihead attention.
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def original_forward(self, query, key, value, mask=None):
        """"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2

        TODO: review transpose operations, not sure this issue is valid or not but reflects
        similar confusion: https://github.com/harvardnlp/annotated-transformer/issues/118.
        """
        # return self.original_forward(query, key, value, mask)

        if mask is not None:
            # Same mask applied to all h heads. Unsqueeze inserts a new
            # dimension a single empty entry at passed index. Example: foo =
            # [2,3]  => foo.unsqueeze(0) = [[2,3]] size=(1,2),  and
            # foo.unsqueeze(1) = [[2],[3]] size=(2x1)
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k.
        # Here, each query, key, and value matrix has associated weights and is
        # run through a linear model, as shown in many tutorials, resulting in
        # the final query, key, and value matrices.

        # query, key, value = [
        #     lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #     # TODO: this appears to be a bug in the original paper source: four
        #     # linear layers are initialized, but only three are used via zip
        #     # which only ranges over the three query, key, value items. Zip only
        #     # iterates up to the smaller of two iterables.
        #     for lin, x in zip(self.linears, (query, key, value))
        # ]
        #
        # The above is a compact way to write these linear ops on the heads, but
        # enumeration per below allows tracking the algebraic dimensions.

        global logger

        # W_q * q   whose dimensions are   (d_model x d_model) * (b x max_padding x d_model)
        # More precisely, W_q in calculations is (d_output x d_input) and its input will
        # be transformed to (d_input x b * max_padding).
        W_q = self.linears[0]
        logger.info(
            f"W_q={W_q.weight.size()} query={query.size()} nbatches={nbatches} d_k={self.d_k}"
        )
        # W_q * q   whose dimensions are   (d_model x d_model) * (b x max_padding x d_model)
        #
        # Note that for torch linear layers, the size constraint per matrix multiplication is
        # per the last dimension, d_model. So the above math is misleading if intepreted per
        # textbook linear algebra. The linear size (d_model x d_model) per implementation is
        # (out_features, in_features) respectively for W. The input tensor of size (b x seqlen x d_model)
        # is internally reshaped to (b * seqlen x d_model). All multiplication per the linear
        # layer and input tensors are then (b * seqlen x d_model) * (d_model x d_model), resulting
        # in (b * seqlen x d_model), where technically this "d_model" is d_model_output_features,
        # which just happens to match (512) of input features. The internal bias vector is size
        # d_output (so again, 512 of d_model) and is added to the output. However, it too is reshaped
        # or rather re-applied to all b segments of output, (b x 512). The simplest way to look at
        # this is simply that the leading dimension is assumed to the batch size, for which the
        # formal mult ops are simply repeated for these dimensions, as the bias vector addition
        # demonstrates most simply.
        #
        # Small example:
        # lin = Linear(input_features, output_features)
        # # input need only agree with lin per input_features
        # out = lin(torch.rand(32, input_features))
        # # out size is (32 x 512)
        # out = lin(torch.rand(128, 32, input_features))
        # # out size is (128 x 32 x 512)
        # out.view(128, -1, 8, 512 // 8)
        # # view is
        #
        #  Wq is (b x max_padding x d_out_features)   from   Wq (32 x 72 x 512)   and   q_out (32 x 72 x 8 x 64)
        Wq = W_q(query)
        # -1 tells torch to calculate the size of that dimension per the others. Hence
        # this results in the q_out dim listed above, where only the last dimensions
        # changes to 8 x 64, effectively meaning that each d_model vector is just
        # broken into 8 different sections with independent weights.
        q_out = Wq.view(nbatches, -1, self.h, self.d_k)
        logger.info(f"Wq={Wq.size()}  q_out={q_out.size()}")
        query = q_out.transpose(1, 2)
        logger.info(f"query={query.size()}")

        # W_k * k   whose dimensions are   (d_model x d_model) * (d_model x)
        W_k = self.linears[1]
        logger.info(f"W_k={W_k.weight.size()} key={key.size()} nbatches={nbatches}")
        Wk = W_k(key)
        k_out = Wk.view(nbatches, -1, self.h, self.d_k)
        logger.info(
            f"Wk={Wk.size()}  k_out={k_out.size()} (where k_out size is (W_k*k).view(nbatches, -1, self.h, self.d_k))"
        )
        key = k_out.transpose(1, 2)
        logger.info(f"key={key.size()} (via k_out.transpose(1, 2))")

        # W_v * v   whose dimensions are   (d_model x d_model) * (d_model x)
        W_v = self.linears[2]
        logger.info(f"W_v={W_v.weight.size()} value={value.size()} nbatches={nbatches}")
        Wv = W_v(value)
        v_out = Wv.view(nbatches, -1, self.h, self.d_k)
        logger.info(f"Wv={Wv.size()}  v_out={v_out.size()}")
        value = v_out.transpose(1, 2)
        logger.info(f"value={value.size()}")

        # 2) Apply attention on all the projected vectors in batch. The second
        # return value self.attn (stored here for visualization) contains the
        # output of softmax and dropout (if any) prior to multiplication by the
        # Value matrix. Thus it contains the relatedness scores of queries and
        # keys, before multiplication/transformation by V back into the model
        # dim-space.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        logger.info(f"x={x.size()} self.attn={self.attn.size()}")

        # 3) "Concat" using a view and apply a final linear.
        #
        # contiguous() ensures that underlying storage of the tensor is
        # contiguous, despite previous tranpose and other view operations; it
        # returns the original tensor if no transforms have been applied, and a
        # copied new tensor otherwise.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        final_out = self.linears[-1](x)
        # final_out = (32 x 72 x 512)
        logger.info(f"x reshaped={x.size()} final_out={final_out.size()}")

        return final_out


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    """
    Embeddings are used to convert input/output tokens to vectors of size
    d_model.

    The weights are multiplied by sqrt(d_model)
    """

    def __init__(self, d_model: int, vocab: int):
        super(Embeddings, self).__init__()
        self.lookup_table = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lookup_table(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.

    The position encodings are 'frozen' embeddings, in that they have
    requires_grad set to false, and no weights are learned during training. They
    merely add positional information--which is pretty interesting.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


def make_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: int = 0.1,
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # FF network layer maps d_model -> d_ff hidden layer -> d_model.
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        # Sequential: modules are added to it in the order passed in the
        # constructor. The ``forward()`` method of ``Sequential`` accepts any
        # input and forwards it to the first module it contains. It then
        # "chains" outputs to inputs sequentially for each subsequent module,
        # finally returning the output of the last module.
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        generator=Generator(d_model, tgt_vocab_size),
    )

    # This was important from their code. Initialize parameters with Glorot /
    # fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Not really the place to do this, but crrently the most central place
    # to display model characteristics.
    torchinfo.summary(model)

    return model


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    global logger
    logger.info("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


# show_example(run_tests)


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words.

        TODO: there is a reported bug in this code, need to review. https://github.com/harvardnlp/annotated-transformer/issues/137
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            logger.info(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# This corresponds to increasing the learning rate linearly for the first
# $warmup\_steps$ training steps, and decreasing it thereafter proportionally to
# the inverse square root of the step number.  We used $warmup\_steps=4000$.
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function to avoid zero raising
    to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for _, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


class LabelSmoothing(nn.Module):
    """
    Regularization: implement label smoothing.

    During training, we employed label smoothing of value $epsilon_{ls}=0.1$
    [(cite)](https://arxiv.org/abs/1512.00567). This hurts perplexity, as the
    model learns to be more unsure, but improves accuracy and BLEU score.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def example_label_smoothing():
    """
    Show how confidence is distributed among words.
    """
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color("target distribution:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


# show_example(example_label_smoothing)


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


# show_example(penalization_visualization)


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


def greedy_decode(model: EncoderDecoder, src, src_mask, max_len, start_symbol):
    """greedy_decode encodes the entire input sequence and then repeatedly samples
    the argmax term at each time step from the decoder.
    """

    global logger
    memory = model.encode(src, src_mask)
    # memory.size()=torch.Size([32, 72, 256]), src=torch.Size([32, 72]) src_mask=torch.Size([32, 1, 72])
    logger.warning(
        f"memory.size()={memory.size()}, src={src.size()} src_mask={src_mask.size()}"
    )

    # Initially ys is empty except for the start symbol. With the encoder loaded
    # with input, we can then predict one token at a time, concatenating each
    # subsequent token onto ys and repeating.
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        # At each time step, mask all subsequent terms to prevent the model looking ahead.
        ys_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        # ys=torch.Size([1, 2]) ys_mask=torch.Size([1, 2, 2])
        # TODO: this is where to implement beam search. Rather than only look
        # one word ahead, probe using beam search for subsequences of max value.
        out = model.decode(memory, src_mask, ys, ys_mask)
        logger.warning(f"ys={ys.size()} ys_mask={ys_mask.size()} out={out.size()}")

        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        logger.info(f">>> next_word size: {next_word.size()}")
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        logger.info(f"Next ys: {ys.size()}")
    return ys


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


def load_tokenizers() -> Tuple[Language, Language]:

    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def build_en_vocabulary(
    train_iter: TGenerator[Tuple[str, str], None, None],
    val_iter: TGenerator[Tuple[str, str], None, None],
    spacy_en: Language,
) -> Vocab:
    """
    Returns a Vocab object for mapping tokens to indices.

    @train_iter: An iterator producing (str,str) tups, where the first item is
    the input sentence and the second is the target output. @val_iter: The
    validation iterator, the same type as @train_iter. @spacy_en: The spacy
    english model.
    """

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building English Vocabulary ...")
    vocab = build_vocab_from_iterator(
        yield_tokens(itertools.chain(train_iter, val_iter), tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab.set_default_index(vocab["<unk>"])

    return vocab


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


# if is_interactive_notebook(): # global variables used later in the script
#     spacy_de, spacy_en = show_example(load_tokenizers) vocab_src, vocab_tgt =
#     show_example(load_vocab, args=[spacy_de, spacy_en])


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    """
    Batching matters a ton for speed. We want to have very evenly divided
    batches, with absolutely minimal padding. To do this we have to hack a bit
    around the default torchtext batching. This code patches their default
    batching to make sure we search over enough sentences to find tight batches.
    """

    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )

        global logger
        if max_padding - len(processed_src) < 0:
            logger.warning("Overwrite occurs for negative value of a padding - len")
        logger.info
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        if max_padding - len(processed_tgt) < 0:
            logger.warning("Overwrite occurs for negative value of a padding - len")
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


"""
JW: the Ai summary is actually pretty good for explaining some of this:

    Why you might see get_stoi() in spaCy examples Some online tutorials for
    deep learning with spaCy and PyTorch may show code where get_stoi() is used.
    In these cases, spaCy is used only for its tokenization. The resulting
    tokens are then passed to a torchtext Vocab object, which has its own
    methods for converting strings to integers for deep learning models.

    For example:

        import spacy from torchtext.vocab import build_vocab_from_iterator

        # Example using spaCy for tokenization and torchtext for vocabulary nlp
        = spacy.load("en_core_web_sm") corpus = ["Hello world!", "This is a
        test."] tokenized_corpus = [[token.text for token in nlp(text)] for text
        in corpus]

        # Build the vocabulary using torchtext vocab =
        build_vocab_from_iterator(tokenized_corpus, specials=["<unk>"])

        # Now you can use the `get_stoi()` method on the torchtext vocab object
        stoi = vocab.get_stoi() print(stoi)

"""


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=("de", "en"))

    # DistributedSampler needs a dataset len()
    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def read_novel_sentences(fpath: str):
    lines = []
    # Read all lines, omitting empty lines and those starting with "CHAPTER"
    with open(fpath, "r", encoding="utf8") as ifile:
        lines = [
            line.strip()
            for line in ifile.readlines()
            if len(line.strip()) > 0 and not line.strip().startswith("CHAPTER")
        ]
    # Common delimiters in huckfinn
    sentence_delimiters = set(["?", ".", ":", "!", ";"])
    # Wastefully join the entire sequence so it can be split on sentence
    # delimiters more simply.
    unsafe_huge_string = " ".join(lines)
    del lines

    sentences = []
    start = 0
    end_index = 1
    while start < len(unsafe_huge_string) and end_index < len(unsafe_huge_string):
        # Consume input until we find the start of next sentence.
        # while unsafe_huge_string[start] in sentence_delimiters:
        #     start += 1
        #     endIndex = start + 1
        if unsafe_huge_string[end_index] in sentence_delimiters:
            sentence = unsafe_huge_string[start:end_index].strip()
            start = end_index + 1
            end_index = start + 1
            if len(sentence) > 0:
                sentences.append(sentence)
        end_index += 1

    del unsafe_huge_string
    return sentences


def clean_novel_sentences(lines: List[str]) -> List[str]:
    """
    Given a list of sentences from some novel, which are language sequences
    broken at sentence delimiters like "?" and ".", we still need to do some
    additional cleaning. There are probably better libraries for this by now,
    but for now this is just ad hoc cleaning. Lowercase and remove all
    punctuation; this way the model will just learn language relations. It would
    be nice to additionally tokenize with wordpieces after this step.
    """
    # FUTURE: leverage libraries or use string.punctuation and other builtins.
    #
    # TODO: it might be interesting to leave command and ending chars, to see
    # how the model concludes with 'voice'.
    characters_to_remove = "[.,!]()?\"“”'‘’`{}*:"
    t_table = dict()
    for c in characters_to_remove:
        t_table[c] = None
    # Some characters should be replaced with a space, like "-" in "and then—as
    # I was saying—he went to the farm..."
    t_table["—"] = " "
    t_table["-"] = " "

    t_table = str.maketrans(t_table)

    return [line.lower().translate(t_table) for line in lines]


def get_novel_sentence_iters(
    fpath: str, split: float = 0.8
) -> Tuple[
    TGenerator[Tuple[str, str], None, None], TGenerator[Tuple[str, str], None, None]
]:
    """
    From fpath, read all lines, splits on any of {!?.:}, preserving these at the
    end of sequences, and yields each sentence in this manner as a tuple pair.
    The return type just matches previous requirements, which were for
    translation tasks where tuple (str,tgt) pairs were each in separate
    languages.

    The lines are shuffled then split into train and validation/test.

    This returns the sentences as-is, no additional parsing. The tokenizer will
    take care of that, plus any other hooks to condition the text.
    """
    lines = read_novel_sentences(fpath)
    lines = clean_novel_sentences(lines)
    with open("train_lines.txt", "w+") as ofile:
        for line in lines:
            ofile.write(line + "\n")

    random.shuffle(lines)
    splitIndex = int(len(lines) * split)
    train_lines = lines[0:splitIndex]
    val_lines = lines[splitIndex:]

    return ((line, line) for line in train_lines), ((line, line) for line in val_lines)


def create_seq_dataloaders(
    novel_path,
    device,
    vocab_src,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
) -> Tuple[DataLoader, DataLoader]:
    """
    @novel_path: The path to a novel, such as the gutenberg huckfin utf8 novel,
    whose sentences can be used as training sequences. The novel should be utf8,
    line based, with sentences delimited by periods. This function chops the
    novel into sentences as training data, where the input sentence is both the
    source and the target.

    Objective: ensure that this function creates dataloaders the same as the
    create_dataloaders method, and can load src/tgt examples for a single
    language and text.

    Creates and return sequential dataloaders, for which the source and target
    sequences are identical, i.e. for prediction tasks.
    """

    # Returns tokenized text. Example:
    # - in:  [token.text for token in  spacy_en.tokenizer("foo Bar")]
    # - out: ['foo', 'Bar']
    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_en,
            tokenize_en,
            vocab_src,
            vocab_src,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    """
    This is the spot at which text sequences are injected, and therefore the
    point at which to modify the problem structure from translation, to
    prediction, etc. The test_iter was unused in the original example code, so
    I've omitted it here as well.

    This matches the original call:
        train_iter, valid_iter, test_iter = datasets.Multi30k(
            language_pair=("de", "en"))

    For which:
            :return: DataPipe that yields tuple of source and target sentences
        Number of lines per split: - train: 29000 - valid: 1014 - test: 1000

    The original iterators (train_iter, valid_iter) yield tuples of (str,str),
    for example: ('Zwei Chinesen stehen an einer Wandtafel.', 'Two chinese
    people are standing by a chalkboard.')
    """
    # TODO: make these iters, not funcs.
    train_iter, valid_iter = get_novel_sentence_iters(novel_path)

    train_iter_map = to_map_style_dataset(train_iter)
    # DistributedSampler for sampling the training data for specific effects.
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    # Data loader combines a dataset and a sampler, and provides an iterable
    # over the given dataset. There is good DI here, this abstraction wraps all
    # batching, sampling, and conversion from vocab to embedding vectors.
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


# Adapted from train_worker, this version is my own for training on
# straightforward language prediction: given sequences, encode/decode them in
# training, and at generation time generate one token at a time.
#
# No gpu was used here, because my laptop doesn't have one and this is just to
# develop the code.
def my_train_worker(
    vocab: Vocab,
    spacy_en: Language,
    config: dict,
) -> EncoderDecoder:
    pad_idx = vocab["<blank>"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]

    print(
        f"Training params: pad_idx={pad_idx} d_model={d_model} num_layers={num_layers} num_epochs={num_epochs} vocab-len={len(vocab)}"
    )

    model = make_model(len(vocab), len(vocab), N=num_layers, d_model=d_model)
    module = model
    is_main_process = True

    # LabelSmoothing provides regularization. See and run
    # example_label_smoothing.
    criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=0.1)

    train_dataloader, valid_dataloader = create_seq_dataloaders(
        config["data_path"],
        torch.device("cpu"),
        vocab,
        spacy_en,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
        is_distributed=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    # Sets the learning rate of each parameter group to the initial lr times a
    # given function. When last_epoch=-1, sets initial lr as lr.
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
        verbose=True,
    )
    train_state = TrainState()

    for epoch in range(num_epochs):
        model.train()
        print(f"CPU Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            file_path = "%s_%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)

    if is_main_process:
        file_path = "%s_final.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)

    return model


def my_generation(model: EncoderDecoder, vocab: Vocab):
    src_text = "the widow she cried over me and called me a poor lost lamb".split(" ")
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# The original train_worker, using gpus and built to train on translation.
def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))

    return model


def my_load_trained_model(
    vocab_src: Vocab, vocab_tgt: Vocab, config: dict, model_path: str
):
    if not exists(model_path):
        raise Exception(f"Model path not found: {model_path}")

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        h=config["h"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(model_path))

    return model


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=7,
    pad_idx=2,
    eos_string="</s>",
    max_len=72,
):
    global logger
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        logger.info(
            f">> rb.src={rb.src} rb.src.size()={rb.src.size()} rb.src_mask.size()={rb.src_mask.size()}"
        )
        # rb.src.mask={rb.src_mask}  rb.src_mask.size()={rb.src_mask.size()}")
        # greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))
        ys = greedy_decode(
            model,
            rb.src,
            rb.src_mask,
            max_len,
            vocab_src["<s>"],
        )
        model_out = ys[0]
        logger.warning(f"ys={ys.size()} model_out={model_out.size()}")
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
        exit(1)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    """Convert a dense matrix to a data frame with row and column indices."""
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7] layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    # batch object for the final example
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


# show_example(viz_encoder_self)


def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


# show_example(viz_decoder_self)


def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


def train_huckfinn():
    """
    Objective: to learn attention/transformer models, I want to train on
    Huckleberry Finn. A lot of this is just retreading old ML/NLP projects and
    knowledge and to begin catching up to progress since the Attention is All
    You Need paper. The original transformer was used for translation, mapping
    english training data to german translations. The assumption here is that
    the same data can be used for input and output in order to train a
    prediction model for prediction/generation tasks instead of translation.
    """

    # Read and convert Huckfinn to dataset reqs.
    huckLines = []
    with open("./data/huckfinn_utf8.txt", "r", encoding="utf8") as ifile:
        hucklines = [
            line.strip()
            for line in ifile.writelines()
            if len(line.strip()) > 0 and not line.startswith("CHAPTER")
        ]

    # Train

    # Generate
