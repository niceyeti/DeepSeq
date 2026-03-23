import os
import itertools
import random
import logging
import math
import copy
import time

# FUTURE: would like to dump python dataclass usage and use pydantic instead.
from dataclasses import dataclass
from pydantic import BaseModel
from typing import Generator as TGenerator, Tuple, List, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
from torch.optim.lr_scheduler import LambdaLR
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
from torch.utils.data.distributed import DistributedSampler

import torchinfo
import pandas as pd
import altair as alt
import spacy
from spacy import Language

from transformer_config import TransformerConfig

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
log = logging.getLogger()


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
    "Generator defines a standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """forward supports inference at production time, after training. This
        is called to project the output of the trained model and subsequently
        run through log-softmax.

        Args:
            * x: a tensor of size ()
        """

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
        """forward takes the x input to encode and its mask.

        Args:
            * x: a tensor of size (b x seq_len x d_model)
            * mask: a tensor of size (b x 1 x seq_len)
        """

        log.info(f"Encoder.forward: x={x.size()}   mask={mask.size()}")

        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """LayerNorm norms its input by centering the mean of its input and dividing
    by the st-dev, then wrapping this in learnable weights and biases as its
    output.
    """

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
    """Normalizes input, runs it through the sublayer, then dropout, then and adds
    input. For code simplicity the norm is first as opposed to last. Eq: x +
    dropout(sublayer(norm(x)). Recall that Dropout randomly zeroes components of
    its input with probability p, as a regularization technique to discourage
    neurons from co-adaptations on training data.

    NOTE: Dropout and similar techniques are training-time constructs, and will
    negatively impact performance at prod-time. Ensure they are shut off by
    calling model.eval() and `with torch.no_grad()` before inferencing!
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
        """Follow Figure 1 (left) for connections.

        Args:
            * x: tensor of size (b x seq_len x d_model)
            * mask: tensor of size (b x 1 x seq_len)
        """
        # global log
        log.info(f"EncoderLayer forward:  x dim {x.size()}  mask dim {mask.size()}")
        # Encode the input through attention function, then adding/norming
        # through sublayer.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        log.info(f"encoder x dim: {x.size()}")
        # Run the encoded output through feed-forward layer for final output.
        x_final = self.sublayer[1](x, self.feed_forward)
        log.info(f"Encoder x_final dim: {x_final.size()}")

        return x_final


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
        tgt-self and src attention."""
        # Encode the target using self-attention and masking to prevent
        # look-ahead, before adding/norming this to the initial target input via
        # the sublayer.
        masked_tgt_attn = self.sublayer[0](
            target, lambda tgt: self.self_attn(tgt, tgt, tgt, tgt_mask)
        )
        # Encoder the target again, this time taking the encoded input as the
        # query vector by which to learn translation features tgt*input, then
        # again adding/norming this to the initial layer input.
        src_attn = self.sublayer[1](
            masked_tgt_attn, lambda src: self.src_attn(src, memory, memory, src_mask)
        )
        # Finally, run the encoded input through a feed forward network.
        return self.sublayer[2](src_attn, self.feed_forward)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """forward

        Args:
            * x: tensor of size (b x (seq_len-1) x d_model)
            * memory: tensor of size (b x seq_len x d_model) taken from the
              final encoder output for seq2seq transformer models
            * src_mask: tensor of size (b x 1 x seq_len)
            * tgt_mask: tensor of size (b v (seq_len-1) x (seq_len-1))
        """

        log.info(
            f"Decoder.forward  x={x.size()}  memory={memory.size()}  src_mask={src_mask.size()} tgt_mask=P{tgt_mask.size()}"
        )

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderOnlyLayer(nn.Module):
    """DecoderOnlyLayer is identical to DecoderLayer but omits the src-mask and
    encoder input. This layer definition is solely for decoder-only models, and
    cannot be used for encoder-decoder models from the original paper. This
    decoder has only self-attention and no src-attention.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderOnlyLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # There are 2 fully connected sublayers in the decoder-only layer: after
        # self-attention, and finally after decoder output.
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, target, tgt_mask):
        """The decoder runs self-attention only."""
        # Run target input through self-attention, with target mask to prevent
        # look-ahead, before passing this along to the sublayer to add/norm the
        # attention output to the initial target input.
        x = self.sublayer[0](target, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Finally, run the encoded target through the feed forward layer before
        # adding/norming a final time through the sublayer.
        return self.sublayer[1](x, self.feed_forward)


class DecoderOnly(nn.Module):
    """DecoderOnly is just the counterpart to Decoder, for mnemonic-sake.
    A block of plain decoders without an encoder (aka "memory") input or
    src-mask only takes the x input and target mask.

    TODO: rename "DecoderOnly*" classes once these factor out well. These were
    named simply to distinguish them from the original Decoder classes built for
    the encoder-decoder model, but these need less-weird names.
    """

    def __init__(self, layer: DecoderOnlyLayer, N: int):
        super(DecoderOnly, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)


# NOTE: under construction. I need to revisit this later, as I've kludged the connections
# between decoder layers, particularly need to correct the 'memory' portion. Refer to
# the EncoderDecoderModel's usage of 'memory', I have made some simply mistakes here but
# need to diagram the architecture to fix this up correctly.
class DecoderModel(nn.Module):
    """DecoderModel is a complete decoder-only model which gets rid of the encoder
    used in the original transformer paper. This model takes as input the same
    input format as the full architecture's encoder model; it then applies
    multiple self-attention layers, but note that this omits the src-attention
    by which the original full-architecture decoder takes the encoder's output.
    Thereafter the only distinction from an encoder-only architecture is that a
    decoder masks input such that it operates auto-regressively and cannot peek
    ahead.
    """

    def __init__(
        self,
        decoder: DecoderOnly,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        super(DecoderModel, self).__init__()
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, tgt, tgt_mask):
        """Process masked target sequences. Note how this takes no encoder
        input, but takes the tgt_mask to ensure that training is
        auto-regressive.

        @tgt: b x max_padding  (LongTensor)
        @tgt_mask: b x (max_padding-1) x 1 (BoolTensor)

        The input @tgt is simply an integer lookup; the Embedding layer is used
        to lookup each of these vectors from the model. I don't know why the
        masks are max_padding-1, but presumably because there is always at least
        one word.

        Per masking, there are basically three cases which should be detailed
        separately where used in the code:
            1) causal masking to ensure the decoder learns to function
               autoregressively by depriving it of information on future output
               tokens
            2) padding token masking to ensure that padding tokens are not
               learnt
            3) other masking to perform custom training to attend to specific
               tokens for domain oriented tasks
        """
        log.info(f"dec tgt: {tgt.size()}  {tgt.type()}")
        log.info(f"dec tgt_mask: {tgt_mask.size()}  {tgt_mask.type()}")

        return self.decode(tgt, tgt_mask)

    def decode(self, tgt, tgt_mask):
        """decode runs the entire decoder layers.

        Args:
            * tgt: size varies according to how much previous context (previous
              words, 'c') is loaded, (1 x c).
            * tgt_mask: size is coupled to tgt size, and is a triangular matrix
              whose elements above the diagonal are true. The size per @tgt is
              (1 x c x c).
        """
        # global log
        log.info(
            f"decoder-model layer: tgt.size()={tgt.size()} tgt_mask.size()={tgt_mask.size()}"
        )

        return self.decoder(self.tgt_embed(tgt), tgt_mask)

    def ensure_inference_mode(self):
        if self.training:
            log.warning(
                "Model.training true in beam_decode. This is generally incorrect, "
                + "as you should call model.eval() and 'with torch.no_grad()` "
                + "before inferencing."
            )


class EncoderDecoderModel(nn.Module):
    """A standard Encoder-Decoder architecture for translation tasks, also
    referred to as an attention-based sequence-to-sequence transformer in
    Bishop's Deep Learning. In this architecture, the original, the queries come
    from the previous decoder layer, and the memory keys and values come from
    the output of the encoder.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Sequential,
        tgt_embed: nn.Sequential,
        generator: Generator,
    ):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Process masked src and target sequences. The input is fully encoded
        before being processed by the decoder.

        Args:
            * src: b x max_padding  (LongTensor)
            * tgt: b x (max_padding-1)  (LongTensor)
            * src_mask: b x 1 x (max_padding)  (BoolTensor)
            * tgt_mask: b x (max_padding-1) x 1  (BoolTensor)

        Returns: a tensor of size (b x (seq_len-1) x d_model)

        The input @src and @tgt are both simply integer lookups; the Embedding layer is used
        to lookup each of these vectors from the model.
        I don't know why the masks are max_padding-1, but presumably because there is always at least one word.

        Per masking, there are basically three cases which should be detailed
        separately where used in the code:
            1) causal masking to ensure the decoder learns to function
               autoregressively by depriving it of information on future output
               tokens
            2) padding token masking to ensure that padding tokens are not
               learnt
            3) other masking to perform custom training to attend to specific
               tokens for domain oriented tasks
        """
        # global log
        log.info(
            f"encdec src: {src.size()} tgt: {tgt.size()}"
            + f"  src_mask: {src_mask.size()}  tgt_mask: {tgt_mask.size()}"
        )

        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """encode runs the encoder layers and gets their final output.

        Args:
            * src: b x max_padding  (LongTensor)
            * src_mask: b x 1 x max_padding  (BoolTensor)
        """
        # global log
        log.info(
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
        log.info(
            f"encdec decoder layer: memory.size()={memory.size()} src_mask.size()={src_mask.size()} tgt.size()={tgt.size()} tgt_mask.size()={tgt_mask.size()}"
        )

        out = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

        log.info(f"encdec final out: {out.size()}")

        return out

    def ensure_inference_mode(self):
        if self.training:
            log.warning(
                "Model.training true in beam_decode. This is generally incorrect, "
                + "as you should call model.eval() and 'with torch.no_grad()` "
                + "before inferencing."
            )


class EncoderModel(nn.Module):
    """
    TODO: this class is a work in progress and is untested.

    TODO: cmd line support for encoder-only training, i.e. with '--encoder-only' or other flag.
    See the top-level TODO tasking, as the spec for this was recorded.

    A standard Encoder-only architecture. This simply removes the Decoder from
    EncoderDecoder. Encoders train without a look-ahead mask such that the learned
    representations attend to all other information in the sequence, forward or
    backward.
    """

    def __init__(
        self,
        encoder: Encoder,
        src_embed: nn.Sequential,
        generator: Generator,
    ):
        super(EncoderModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, src_mask):
        """Process masked src and target sequences. The input is fully encoded
        before being processed by the decoder.

        @src: b x max_padding  (LongTensor)
        @src_mask: b x 1 x (max_padding-1)  (BoolTensor)

        The input @src is simply an integer lookup; the Embedding layer is used
        to lookup each of these vectors from the model. I don't know why the
        masks are max_padding-1, but presumably because there is always at least
        one word.

        Per masking, there are basically three cases which should be detailed
        separately where used in the code:
            1) causal masking to ensure the decoder learns to function
               autoregressively by depriving it of information on future output
               tokens
            2) padding token masking to ensure that padding tokens are not
               learnt
            3) other masking to perform custom training to attend to specific
               tokens for domain oriented tasks
        """
        # global log
        log.info(f"encoder.forward src: {src.size()}  {src.type()}")
        log.info(f"encoder.forward src_mask: {src_mask.size()}  {src_mask.type()}")

        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        """encode runs the encoder layers and gets their final output.

        @src:
        @src_mask:
        """
        # global log
        log.info(
            f"enc-only encoder layer: src.size()={src.size()} src_mask.size()={src_mask.size()}"
        )
        return self.encoder(self.src_embed(src), src_mask)

    def ensure_inference_mode(self):
        if self.training:
            log.warning(
                "Model.training true in beam_decode. This is generally incorrect, "
                + "as you should call model.eval() and 'with torch.no_grad()` "
                + "before inferencing."
            )


def subsequent_mask(size):
    """Returns a tensor of size (1 x @size x @size) for masking out subsequent
    positions, thus ensuring the auto-regressive property of decoders. This
    returns an upper triangular matrix of booleans whose diagonal and
    below-diagonal entries are False.

    Returns: a tensor whose elements above the diagonal are False, and both the
    diagonal and below-diagonal elements set to True.

    >>> torch.triu(torch.ones(1,4,4), diagonal=1)
    tensor([[[0., 1., 1., 1.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.]]])

    >>> torch.triu(torch.ones(1,4,4), diagonal=1) == 0
    tensor([[[ True, False, False, False],
            [ True,  True, False, False],
            [ True,  True,  True, False],
            [ True,  True,  True,  True]]])
    """

    # Note: the '1' first dim is inferred by torch to be for batch, i.e.
    # torch.ones((1,2,2)) is '[[[1, 1],[1, 1]]]'
    attn_shape = (1, size, size)
    # Creates an upper triangular matrix whose diagonal is zero; the 'diagonal'
    # param indicates how many diagonals above the main diagonal to shift the
    # upper elements.
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


# TODO: is mask ever None? This check bugs me, remove default of None if possible.
def attention(query, key, value, mask=None, dropout=None, module_name: str = ""):
    """
    Per the masking issue: it might be worth looking at
    https://github.com/harvardnlp/annotated-transformer/issues/137. There is a
    reported issue with make_std_mask. The first-principles definition of
    std-mask has not been reviewed and is the most opaque part of this code.

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

    The inputs to this method are more detailed in their preparation in the
    MultiheadedAttention forward() method. Each query, key, and value argument
    represents each input token embedding sliced into h components and grouped
    per head. To understand the functionality it is best to ignore the batch
    dimension; focus on (h x seq_len x (d_model / h)) and note the indexing into
    this: 0th -> head, 1st -> seq token, 2nd -> components of sliced embeddings.
    In plain english, the first index provides the sequence of embeddings of h1.
    If the original embedding sequence is [a, b, c, d] where each is d_model and
    h is 2, then the 0th index contains [a0, b0, c0, d0] and 1st index contains
    [a1, b1, c1, d1], where a0 is the first half of the a embedding and a1 is
    the second half.

    Now, for the actual dot-product attention operation of these inputs, q =
    [q0, q1, q2, q3] and k = [k0, k1, k2, k3] where each 'x_i' (e.g. q0) is
    dimension (d_model / h), so some vector, and each of these represent
    sequential tokens (as sliced token embeddings). The transpose operation
    ensures proper vector dot producting: let q and k be seq_len x (d_model /
    h); then k must be transposed such that each token slice is dotted,
    satsifying dot(q0, k0). The result score matrix, where '*' denotes dot
    product, is:

        scores =
            q0*k0  q0*k1  q0*k2 ... q1*k0  q1*k1  q1*k2 ... q2*k0  q2*k1  q2*k2
            ... ...

    After masking:

        scores =
            q0*k0   -1e9   -1e9 ... q1*k0  q1*k1   -1e9 ... q2*k0  q2*k1  q2*k2
            ... ...


    Masking then ensures that query tokens cannot be related to key tokens at
    indices greater than the query.

    Args:
        * query: b x h x seq_len x (d_model / h)
        * key:   b x h x seq_len x (d_model / h)
        * value: b x h x seq_len x (d_model / h)
        * mask:  (b x 1 x seq_len) for encoder self-attn, and (b x 1 x
          (seqlen-1) x (seqlen-1)) for tgt cross-attention.
        * dropout: if any
    """
    # Get the dimensionality d_head
    d_head = query.size(-1)
    # Transpose key such that matrix multiplication. The important dimensions,
    # omitting the leading batch dimension, are (head, seq_len, d_head). The
    # matrix multiplication here establishes the relationship between every
    # input token of each input sequence, an O(n**2) op. So mentally now you can
    # also ignore the head dimension, and focus on (seq_len, d_head). The
    # transpose simply arranges sizes for this matrix-multiplication, such that
    # every src token is related with every tgt token, via dot product.
    # Accordingly, the output dimension will be (seq_len, seq_len). Thus,
    # transpose key from (seq_len, d_head) to (d_head, seq_len), which is what
    # this transpose(-2, -1) performs.
    k_t = key.transpose(-2, -1)
    log.info(
        f"{module_name}  In attn: d_k={d_head} q={query.size()} k={key.size()} k_t=k.transpose(-2,-1)={k_t.size()} mask={(mask.size() if mask is not None else "NONE")}"
    )

    """
    Scores are (seq_len x seq_len) for each src and tgt sequence (ignoring head
    and batch dimensions). There are two cases, (1) query and keys (and values)
    represent tokens from the same sequence (2) they represent tokens from
    separate sequences. For (1) the matrix multiplication to generate the scores
    relate tokens to eachother, for (2) they relate possibly inter-language
    words to one another.

    * query size: (64, 8, 72, 64) which is (b, h, seq_len, d_model / h)
    * key size: (64, 8, 72, 64) which is (b, h, seq_len, d_model / h)
    * key_t size: (64, 8, 64, 72) which is (b, h, d_model / h, seq_len)
    * scores size: (64, 8, 72, 72) which is (b, h, seq_len_q, seq_len_k)

    When inspecting sizes, it is best to ignore batch (b) and h (num heads), and
    focus only on the sequences of head-chopped feature vectors of size (d_model
    / h).
    """
    scores = torch.matmul(query, k_t) / math.sqrt(d_head)

    if mask is not None:
        # Set masked scores to negative large-numbers, such that their output
        # probs are effectively zero.
        old_shape = mask.shape
        if mask.size()[-1] != scores.size()[-1]:
            mask = mask.view(1, 1, 1, -1)
            log.error(
                f"MASK MODIFICATION {module_name}  mask size {old_shape} != scores size {scores.size()}, reshaped to {mask.size()}. This is currently needed during inference because ys has no batch dim."
            )
            raise ValueError(
                "Exiting. Review the previous error and sizes to understand why the obsolete mask view() call was required."
            )

        log.info(
            f"{module_name}  q={query.size()} k={key.size()} k_t={k_t.size()} v={value.size()}"
            + f"  scores={scores.size()} mask={mask.size()} mask_old_shape={old_shape}"
        )

        """masked_fill: this functionality is given online from this source example.
            [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]]

        You want to change the values at [0,0] and [1,1] to 9.9, as:

            [[9.9, 2.0, 3.0],
            [4.0, 9.9, 6.0]]

        Create a mask, 'msk':
            
            [[0, 1, 1],
            [1, 0, 1]]

        Then call masked_fill():

            result = source.masked_fill(msk == 0, 9.9)

        Here, mask is a matrix whose entries above the diagonal are set to False.
        Despite the developer rule to use assertive values (not 'False'), what this
        says is that mask the False entries to a very large negative value before
        the softmax such that after softmax, these entries are effectively zero,
        and therefore excluded from modeling. The indices of the scores matrix
        represent sequence indices: the row indices are src sequence indices, and
        the col indices are tgt indices. The diagonal and elements below it allow
        information to be modeled for a src token's relationship with tgt elements at
        the same index.
        """

        # Mask the scores. There are two cases, (1) the mask applies to padding
        # tokens (2) the mask applies to above diagonal key tokens to prevent
        # the model from looking ahead. The mask may be (1 x col) sized, i.e.
        # for src-attn masking; in this case the mask is simply broadcast to all
        # rows of the resulting matrix. For (2) the mask is upper-triangular
        # such that above-diagonal elements are set to -1e9, and thereby no
        # qi has a score for any kj where j>i, and thereby the query tokens
        # cannot be related to key tokens that are further in the sequence.
        #
        # TODO: for the self-attn case, the mask is sized (b x 1 x 1 x seqlen),
        # i.e. ([64, 1, 1, 72]). This broadcasts this mask over all columns of
        # padding tokens defined by the mask; this seems incorrect, as the
        # scores matrix represents the sequence in both the row and col
        # direction, therefore the mask ought to mask columns after the first
        # pad-idx column, as well as rows after that same pad-idx. I need to
        # justify why this is not done.
        scores = scores.masked_fill(mask == False, -1e9)

    # Apply softmax across the last dimension of the scores. This applies
    # softmax across the columns.
    #
    # The @scores matrix is (seqlen x seqlen) whose rows span the dimension of
    # keys, i.e. the first row is [ q0*k0 q0*k1 q0*k2 ]. Per the paper, softmax
    # is applied to the rows of this matrix, i.e. across the components of k[*].
    # I don't know much the orientation of softmax matters, but note that for
    # the first row with most subsequent tokens masked, the result is [1.0 0.0
    # 0.0 ... 0.0]. Consequently the earlier rows concentrate more of the
    # probability distribution (at the start of each vector) and the later
    # vectors will be longer and have the softmax output more distributed.
    # Perhaps the values matrix and its weights inherently accounts for this.
    # Also note that when subsequent-token (tgt) masking is applied above the
    # diagonal, the first entry in the first row will always be 1.0; this entry
    # could represent the '<s>' start-of-sequence token, but is worth
    # recognizing.
    key_axis = -1
    p_attn = scores.softmax(dim=key_axis)

    # Apply dropout to the score outputs, randomly assigning zero to entries of
    # the matrix.
    if dropout is not None:
        p_attn = dropout(p_attn)

    """Finally, multiply the weighted scores: ignoring heads, batches, and the
    division of d_model by num heads, p_attn is (seqlen x seqlen) and value is
    (seqlen x d_model). As values contains the original sequence embeddings, the
    resulting matrix applies the weighted (and possibly diagonalized) sums of
    the scores to each component of the values matrix. For example the first row
    entries are: [ q0k0v0  (q1k0v0 + q1k1v1) (q2k0v0 + q2k1v1 + q2k2v2) ... ],
    where qikj represents the softmax of qi and kj's dot product. Notice how
    complicated the resulting entries are; but also note that dot-product
    attention is not the only form, such as additive attention. So this isn't
    set in stone, it is a function dependency and an opportunity for hacking
    other representations.
    """
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multihead attention allows the model to jointly attend to information
    from different representation subspaces at different positions. With a
    single attention head, averaging inhibits this. The linear weights on each
    Q, K, V weight matrix are size d_model, but are simply transformed to the
    space of the heads when needed. Formally, the outputs of each head are
    concatenated together and multiplied by a weight matrix W_0 to get the final
    output.

        Multihead(Q,K,V) = Concat(head_1, head_2, ... head_h) * W^0 where head_i
        = Attention(QW_iq, KW_ik, VW_iv) where W_iq in R[d_model x d_k], W_ik in
        R[d_model x d_k], and W_iv in R[h*d_v x d_model] Basically, the matrices
        are the sizes of the original matrices scaled down by h, and such that
        the row/col math works out the same.
    """

    def __init__(self, h, d_model, dropout=0.1, module_name=""):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert (
            d_model % h == 0
        ), f"d_model % h must equal 0; h must divide d_model but got: h={h} d_model={d_model}"
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # Create 3 linear layers for each of Q, K, and V matrices, plus one for
        # the final output, 4 total. These are the weights by which the linear
        # projection is performed for multihead attention.
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self._module_name = module_name

    def original_forward(self, query, key, value, mask=None):
        """This is the original forward implementation which is much more
        compact and elegant, closely resembles the paper definition of
        attention. I exploded the forward implementation simply for some
        required debugging, which makes this much less readable. Even though
        this func is not called, I'm leaving it for reference to the intended
        implementation."""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k.
        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. Transpose converts
        # the multihead-format of the tensor from (b x h x seqlen x (d_model /
        # h)) back to (b x seqlen x d_model).
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)

    def forward(self, query, key, value, mask=None):
        """Implements that multi-headed attention mechanism. For reference,
        in encoder-decoder sequence to sequence models the query vectors come
        from the output of the previous decoder block, whereas the key and value
        vectors are provided by the final output of the entire encoder chain.

        The internal sizes of W_q, W_k, W_v are identical, d_model. However arg
        size differs for the encoder vs the decoder.

        Args:
            * query: tensor of size (b x max_length x d_model)
            * key: tensor of size (b x max_length x d_model)
            * value: tensor of size size (b x max_length x d_model)
            * mask: tensor of size
        """

        if mask is not None:
            # The same mask is applied to all h heads. Unsqueeze inserts a new
            # dimension a single empty entry at the head index. Example: foo =
            # [2,3]  => foo.unsqueeze(0) = [[2,3]] size=(1,2),  and
            # foo.unsqueeze(1) = [[2],[3]] size=(2x1). The [1] index is the
            # index of the head content, i.e. (b x h x seqlen x (d_model / h)).
            mask = mask.unsqueeze(1)
        q_batch_size = query.size(0)
        k_batch_size = key.size(0)
        v_batch_size = value.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k.
        # Here, each query, key, and value matrix has associated weights and is
        # run through a linear model, as shown in many tutorials, resulting in
        # the final query, key, and value matrices.

        # Note: original compact code setup of the linear layers. The zip looks like
        # a bug as there are four linear layers but zip iterates only the smaller
        # collection (len 3), thus omitting the fourth layer. However the last layer
        # is used in the final output further below.
        #
        # query, key, value = [
        #     lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #     for lin, x in zip(self.linears, (query, key, value))
        # ]
        #
        # The above is a compact way to write these linear ops on the heads, but
        # enumeration per below allows tracking the algebraic dimensions.

        # W_q * q   whose dimensions are   (d_model x d_model) * (b x max_padding x d_model)
        # More precisely, W_q in calculations is (d_output x d_input) and its input will
        # be transformed to (d_input x b * max_padding).
        W_q = self.linears[0]
        log.info(
            f"{self._module_name}  W_q={W_q.weight.size()} query={query.size()} nbatches={q_batch_size} d_k={self.d_k}"
        )
        # q * W_q   whose dimensions are   (d_model x d_model) * (b x max_padding x d_model)
        #
        # Note that for torch linear layers, the size constraint per matrix multiplication is
        # per the last dimension, d_model. So the above math is misleading if interpreted per
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
        # -1 tells torch to calculate the size of that dimension per the others.
        # Hence this results in the q_out dim listed above, where only the last
        # dimensions changes to 8 x 64, effectively meaning that each d_model
        # vector is just broken into 8 different sections with independent
        # weights.

        """
        By omitting the first dimension, the batch, we can get a better view of what is happening here
        by looking at only a single training example, its vectors, and weights.

        Here, r represents a single training example sequence, of two tokens, each of whose embedding dim=4.
        The first array listed in r represents the first token's embedding vector, and the second array
        is the second token's embedding vector.

        >>> r = torch.rand((2,4))
        >>> r
        tensor([[0.0330, 0.0824, 0.2627, 0.8666],
                [0.3143, 0.1763, 0.8792, 0.0755]])

        Reconfiguring with view(-1,2,2) is a representation of the equivalent view(-1, h, d_model / h).
        As shown, the components of the embedding vectors are sliced into equal portions; for example,
        [0.0330, 0.0824, 0.2627, 0.8666] -> [[0.0330, 0.0824], [0.2627, 0.8666]]. Already note the
        intended outcome that each head looks at different portions of embeddings. The 0th element
        or r_v contains the chopped slices of the first token embedding.

        >>> r_v = r.view(-1,2,2)
        >>> r_v
        tensor([[[0.0330, 0.0824],
                [0.2627, 0.8666]],

                [[0.3143, 0.1763],
                [0.8792, 0.0755]]])

        This is the tough part.
        The 0th dimension of the new view contains each token's embedding (sliced);
        the 1st dimension indexes into each head's slice of tokens; the 2nd and last
        dimension represent indices into the components of each of these slices.
        Transposing over (0,1) groups all of the first head's components together,
        the second head, and so on for all heads. This gives h groups (0th dimension),
        each of size seq_len (1st dimension), and each sized d_h (2nd dimension).
        As described, note how the first group of arrays are the h1's components,
        and the second group are h2's components.
        
        tensor([[[0.0330, 0.0824],
                [0.2627, 0.8666]],

                [[0.3143, 0.1763],
                [0.8792, 0.0755]]])
                

        >>> r_v_t = r_v.transpose(0,1)
        >>> r_v_t
        tensor([[[0.0330, 0.0824],
                [0.3143, 0.1763]],

                [[0.2627, 0.8666],
                [0.8792, 0.0755]]])






        Simulates a single training example of len 2 and d_model=4
            >>> r = torch.rand((2,4))
            >>> r
            tensor([[0.5714, 0.5531, 0.1266, 0.2416],
                    [0.4489, 0.8925, 0.2747, 0.6700]])

        Transposing the seq and d_model components places all of the 
        >>> r_t = r.transpose(0,1)
        >>> r_t
        tensor([[0.5714, 0.4489],
                [0.5531, 0.8925],
                [0.1266, 0.2747],
                [0.2416, 0.6700]])



        # Start with a tensor of size (seq_len x d_model)
        >>> r = torch.rand((1,2,4))
        >>> r
        tensor([[[0.4701, 0.0986, 0.3829, 0.0027],
                [0.9231, 0.0034, 0.1877, 0.3378]]])

        # Running transpose(1,2) places the 0th 
        >>> r_t = r.transpose(1,2)
        >>> r_t
        tensor([[[0.4701, 0.9231],
                [0.0986, 0.0034],
                [0.3829, 0.1877],
                [0.0027, 0.3378]]]) 
        
        
        
        """

        q_out = Wq.view(q_batch_size, -1, self.h, self.d_k)
        log.info(f"{self._module_name}  Wq={Wq.size()}  q_out={q_out.size()}")
        query = q_out.transpose(1, 2)
        log.info(f"{self._module_name}  query={query.size()}")

        # W_k * k   whose dimensions are   (d_model x d_model) * (d_model x)
        W_k = self.linears[1]
        log.info(
            f"{self._module_name}  W_k={W_k.weight.size()} key={key.size()} nbatches={k_batch_size}"
        )
        Wk = W_k(key)
        # Regression note: this view change was problematic because of the -1.
        # For an inference task, when query is [1, 8, 1, 64] for a single token
        # sequence, and key is [64, 72, 512] and Wv is [64, 72, 512] for a
        # full-length sequence encoding, then k_out becomes [1, 4608, 8, 64] due
        # to the -1: 'Wk.view(1, -1, 8, 512)'. The resolution is to always pad
        # tgt to full seqlen.
        #
        # TODO: consider error checking sizes on ingress to ensure this
        # regression is logged or raised.
        k_out = Wk.view(k_batch_size, -1, self.h, self.d_k)
        log.info(
            f"{self._module_name}  Wk={Wk.size()}  k_out={k_out.size()} (where k_out size is (W_k*k).view(nbatches, -1, self.h, self.d_k))"
        )
        key = k_out.transpose(1, 2)
        log.info(f"{self._module_name}  key={key.size()} (via k_out.transpose(1, 2))")

        # W_v * v   whose dimensions are   (d_model x d_model) * (d_model x)
        W_v = self.linears[2]
        log.info(
            f"{self._module_name}  W_v={W_v.weight.size()} value={value.size()} v-batches={v_batch_size}"
        )
        Wv = W_v(value)
        v_out = Wv.view(v_batch_size, -1, self.h, self.d_k)
        log.info(
            f"{self._module_name}  Wv={Wv.size()}  v_out={v_out.size()} v_batch_size={v_batch_size}"
        )
        value = v_out.transpose(1, 2)
        log.info(f"{self._module_name}  value={value.size()}")

        # 2) Apply attention on all the projected vectors in batch. The second
        # return value self.attn (stored here for visualization) contains the
        # output of softmax and dropout (if any) prior to multiplication by the
        # Value matrix. Thus it contains the relatedness scores of queries and
        # keys, before multiplication/transformation by V back into the model
        # dim-space.
        x, self.attn = attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            module_name=self._module_name,
        )
        log.info(f"{self._module_name}  x={x.size()} self.attn={self.attn.size()}")

        # 3) "Concat" using a view and apply a final linear.
        #
        # contiguous() ensures that underlying storage of the tensor is
        # contiguous, despite previous tranpose and other view operations; it
        # returns the original tensor if no transforms have been applied, and a
        # copied new tensor otherwise.
        x = x.transpose(1, 2).contiguous().view(q_batch_size, -1, self.h * self.d_k)
        del query
        del key
        del value
        final_out = self.linears[-1](x)
        # final_out = (32 x 72 x 512)
        log.info(
            f"{self._module_name}  x reshaped={x.size()} final_out={final_out.size()}"
        )

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


def make_encdec_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: int = 0.1,
):
    """make_encdec_model builds and returns a model consisting of both an
    encoder and decoder, per the original Attention Is All You Need paper, aka
    an attention-based seq2seq model.
    """
    c = copy.deepcopy
    # FF network layer maps d_model -> d_ff hidden layer -> d_model.
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoderModel(
        encoder=Encoder(
            EncoderLayer(
                d_model,
                MultiHeadedAttention(h, d_model, module_name="encoder-self-attn"),
                c(ff),
                dropout,
            ),
            N,
        ),
        decoder=Decoder(
            DecoderLayer(
                d_model,
                MultiHeadedAttention(h, d_model, module_name="decoder-masked-attn"),
                MultiHeadedAttention(h, d_model, module_name="decoder-srctgt-attn"),
                c(ff),
                dropout,
            ),
            N,
        ),
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

    # Not really the place to do this, but currently the most central place
    # to display model characteristics.
    torchinfo.summary(model)

    return model


def make_decoder_model(
    src_vocab_size: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: int = 0.1,
):
    """TODO: decoder-only model is under construction and untested. This is
    how gpt-style model work I'm told, but really this is more sensible for
    in-language training/prediction tasks, vs. the translation task for which
    encoder-decoder models are intended."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    # FF network layer maps d_model -> d_ff hidden layer -> d_model.
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = DecoderModel(
        decoder=DecoderOnly(DecoderOnlyLayer(d_model, c(attn), c(ff), dropout), N),
        # Sequential: modules are added to it in the order passed in the
        # constructor. The ``forward()`` method of ``Sequential`` accepts any
        # input and forwards it to the first module it contains. It then
        # "chains" outputs to inputs sequentially for each subsequent module,
        # finally returning the output of the last module.
        tgt_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        generator=Generator(d_model, src_vocab_size),
    )

    # This was important from their code. Initialize parameters with Glorot /
    # fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Not really the place to do this, but currently the most central place
    # to display model characteristics.
    torchinfo.summary(model)

    return model


def make_encoder_model(
    src_vocab_size: int,
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
    model = EncoderModel(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        # Sequential: modules are added to it in the order passed in the
        # constructor. The ``forward()`` method of ``Sequential`` accepts any
        # input and forwards it to the first module it contains. It then
        # "chains" outputs to inputs sequentially for each subsequent module,
        # finally returning the output of the last module.
        src_embed=nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        generator=Generator(d_model, src_vocab_size),
    )

    # This was important from their code. Initialize parameters with Glorot /
    # fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Not really the place to do this, but currently the most central place
    # to display model characteristics.
    torchinfo.summary(model)

    return model


def inference_test():
    test_model = make_encdec_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    # global log
    log.info("Example Untrained Model Prediction:", ys)
    return ys


# TODO: should these be moved the unit test?
def encoder_inference_test():
    test_model = make_encoder_model(11, 3)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        prob = test_model.generator(memory[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    # global log
    log.info("Example Untrained Model Prediction:", ys)
    return ys


# TODO: should these be moved the unit test?
def decoder_inference_test():
    test_model = make_decoder_model(11, 3)
    test_model.eval()
    tgt = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    tgt_mask = torch.ones(1, 1, 10)

    memory = test_model.decode(tgt, tgt_mask)
    ys = torch.zeros(1, 1).type_as(tgt)

    for _ in range(9):
        prob = test_model.generator(memory[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(tgt.data).fill_(next_word)], dim=1
        )

    # global log
    log.info("Example Untrained Model Prediction:", ys)
    return ys


def run_tests():
    for _ in range(10):
        inference_test()


class Batch:
    """Object for holding a batch of data with mask during training. Inspect
    and understand this object's attributes carefully, as the tgt input is
    modified to ensure autoregressive training and masking. This can be the most
    difficult code in transformers, but ultimately it implements the downstream
    size requirements determined by the pencil math of tensors in the separate
    attention mechanism in the encoder vs decoder.

    Attributes:
        * src: a tensor of size (b x seq_len)
        * src_mask: a bool tensor of size (b x 1 x seq_len) masks the src input
          padding elements for the given src instance of (b x 1 x seqlen).
        * tgt: a tensor of size (b x seq_len-1). This is tgt output excluding
          the final token (often a mere padding token) used for the encoder.
          NOTE: wherever this is used, confirm on paper that the minus-one size
          is correct.
        * tgt_y: a tensor of size (b x seq_len-1). This is tgt output excluding
          the first token (often a mere padding token), used for the decoder.
          NOTE: wherever this is used, confirm on paper that the minus-one size
          is correct.
        * tgt_mask is the mask derived from make_std_mask and tgt. NOTE: as tgt
          removes one final token, verify that the dimension of tgt_mask is
          correct in its application.
        * ntokens: the number of non-padding tokens, useful for accounting in
          training.

    NOTE: tgt=tgt[:-1] and tgt_y=tgt[1:], offset by one. And for the instance
    that src==tgt, tgt will consist of only src[:-1], chopping the last token
    for tgt, and shifting by one for tgt_y (). Check the src and tgt mask
    implementation below.
    """

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor, pad_id: int = 2):
        """
        Args:
            * src: a tensor of size (b x seq_len)
            * tgt: a tensor of size (b x seq_len)
            * pad: the pad id
        """
        self.src = src

        # src_mask: masks the src input padding elements, i.e. 'tensor([[[ True,
        # True,  True,  ..., False, False, False]], '.
        self.src_mask = (src != pad_id).unsqueeze(-2)

        # TODO: src_mask, tgt, tgt_y and tgt_mask are the most nuanced part of this
        # code and need to be documented; the off by one risks are significant.

        # tgt is the tgt output, EXCLUDING the final token, used for the encoder.
        self.tgt = tgt[:, :-1]
        # tgt_y is the tgt output EXCLUDING the first token, used for the decoder.
        self.tgt_y = tgt[:, 1:]
        # tgt_mask is the mask derived from make_std_mask
        self.tgt_mask = self.make_std_mask(self.tgt, pad_id)
        self.ntokens = (self.tgt_y != pad_id).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad_id: int):
        """Create a mask to hide padding and future words.

        Args:
            * tgt: the tgt tensor of size (b x (seqlen-1))
            * pad: the pad-id

        Returns: a tensor of size (b x (seqlen-1) x (seqlen-1)), whose final two
        dimensions are the upper-triangular boolean mask matrices.

        TODO: there is a reported bug in this code, need to review.
        https://github.com/harvardnlp/annotated-transformer/issues/137
        """
        tgt_mask = (tgt != pad_id).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)

        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
    data_iter: TGenerator[Batch, None, None],
    model: EncoderDecoderModel,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
) -> Tuple[float, TrainState]:
    """Train a single epoch over the entire batch iterator.

    NOTE: remember that Batch offsets tgt by one.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        log.info(f"model.forward  out={out.size()}")
        exit()
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode in ["train", "train+log"]:
            loss_node.backward()
            train_state.step += 1
            # src.shape[0] is size of batch, i.e. #samples
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
        if i % 40 == 1 and mode in ["train", "train+log"]:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            log.info(
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
        for _ in range(20000):
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


def greedy_decode(model: EncoderDecoderModel, src, src_mask, max_len, start_symbol):
    """greedy_decode encodes the entire input sequence and then repeatedly samples
    the argmax term at each time step from the decoder. For the purposes of
    language prediction within a single language, this is effectively a
    inference check on an autoencoder, taking in "the quick brown fox", encoding
    it, and then running the decoder one word at a time to create the same
    sequence. This is roughly similar to summarization, though the exact target
    output is the original sentence.

    NOTE: make sure you call model.eval() before calling this to ensure layers
    like Dropout are disabled!
    """

    # global log
    memory = model.encode(src, src_mask)
    # memory.size()=torch.Size([32, 72, 256]), src=torch.Size([32, 72]) src_mask=torch.Size([32, 1, 72])
    log.info(
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
        # >>>> memory=torch.Size([32, 72, 256]) src_mask=torch.Size([32, 1, 72]) ys=torch.Size([1, 68]) ys_mask=torch.Size([1, 68, 68])
        # print(
        #     f">>>> memory={memory.size()} src_mask={src_mask.size()} ys={ys.size()} ys_mask={ys_mask.size()}"
        # )

        # @out is size (b x cur_len x d_model), where cur_len is the current len
        # of the sequence as it is extended one at a time. Note it is the same
        # size as ys, since we haven't predicted the next word until below.
        out = model.decode(memory, src_mask, ys, ys_mask)
        # WARNING:root:ys=torch.Size([1, 68]) ys_mask=torch.Size([1, 68, 68]) out=torch.Size([1, 68, 256])
        log.info(f"ys={ys.size()} ys_mask={ys_mask.size()} out={out.size()}")

        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        log.info(f">>> next_word size: {next_word.size()}")
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        log.info(f"Next ys: {ys.size()}")
    return ys


def beam_decode(
    model: EncoderDecoderModel, src, src_mask, max_len, start_id, beam_length, pad_id
) -> List[torch.Tensor]:
    """beam_decode is the same as greedy_decode except that we sample the top
    @beam_length outputs as @ys. That is, (1) encode the entire provided input
    sequence, and (2) predict outputs; populate a beam of size @beam_length, then
    run again and replace/add new sequence to the beam that are higher
    probability.

    Returns a list of tensors sorted from most probable to least. No probs are
    included since there is no such requirement. Each tensor contains the word
    ids, by which the actual words can be retrieved from the vocab of the model.

    The definition of probability in this context is the
    probability of the entire sequence, such that the highest probability seq
    has prob P*. Note that each addition to the beam need not increase the
    current length of all predicted sequences. On the first round, the top
    @beam_length output words will always populate the beam; thereafter, some of
    these could remain in the beam if they have arbitrarily higher probability
    than other alternative sequence extensions. The output probabilities
    are one-step probs, not full sequence probabilities.

    1) Initially, populate beam with top-k output words
    2) For each item in beam, extend by one word (d=1), and assign the highest probability
       one to the current word-sequence's probability, and append the word.
        * Modification: for d>1, continue to sample and replace

    """

    # TODO: update me with type info when their structure is understood.
    # Or, remove and replace with tuple if local enough.
    @dataclass
    class BeamItem:
        # The word sequence.
        ys: torch.Tensor
        # The accumulated probability for this sequence.
        prob: float

    model.ensure_inference_mode()

    memory = model.encode(src, src_mask)
    # memory.size()=torch.Size([32, 72, 256]), src=torch.Size([32, 72]) src_mask=torch.Size([32, 1, 72])
    log.info(
        f"memory.size()={memory.size()}, src={src.size()} src_mask={src_mask.size()}"
    )

    # TODO: batchify these ops: intuition says I'm doing this wastefully, but
    # could run inference much faster for a batch of items in the beam at once.
    # Beam search seems extremely amenable to very fast batchification, whereby
    # long beams could be decoded all at once, instead of through iteration.
    batch_size = src.size(0)
    ys = torch.zeros(batch_size, max_len).fill_(pad_id).type_as(src.data)
    # Initialize all batches (there should be only 1) to start-id.
    ys[:, 0] = start_id
    # The beam is a list of tensors and their associated probabilities.
    # Initialized to the start_symbol, with prob chosen such that subsequent
    # probs accumulate correctly.
    beam: List[BeamItem] = [BeamItem(ys=ys, prob=-1.0)]

    # TODO: revise stopping criteria, i.e. when the whole beam's items achieve
    # some length or are all capped with EOS pads.
    for _ in range(20):
        # For each word in the beam, run the decoder to get its top-k most
        # likely next outputs, appending this to the beam. After predicting for
        # all candidate terms and appending their top-k outputs, sort and
        # truncate the beam back to beam-length. K and beam-length are separate
        # parameters, where beam supports total scope and k is effectively the
        # search radius around candidate words.
        #
        # TODO: replace beam with a heap to avoid large beam and sorting.
        #
        # TOD: replace deepcopy with a more efficient pattern when code paths harden. This is cope to prevent the infinite loop of appending
        # to the beam while iterating it.
        next_beam = []
        for beam_item in beam:
            # At each time step, mask all subsequent terms to prevent the model
            # looking ahead.
            ys_mask = Batch.make_std_mask(beam_item.ys, pad_id).type_as(src.data)
            # ys_mask = subsequent_mask(beam_item.ys.size(1)).type_as(src.data)

            # TODO: input to the decoder is given as (1 x seqlen), where b=1.
            # The batch size b could easily be used to support decoding batches
            # of the beam at a time, more efficiently than through iteration.

            # Hybrid beam with greedy depth search:
            # - for each item in the beam
            # - run decoder greedily for d steps, i.e. repeat d times: `next_word = decode(next_word)`
            # - return max_prob, and arg_max sequence for this item to add to beam
            #
            # Other alterations are possible, such as backing up the 'value' of each word per
            # an average over its successors under some strategy, a la Monte Carlo sampling.

            # ys=torch.Size([1, 2]) ys_mask=torch.Size([1, 2, 2])
            # >>>> memory=torch.Size([32, 72, 256]) src_mask=torch.Size([32, 1, 72]) ys=torch.Size([1, 68]) ys_mask=torch.Size([1, 68, 68])
            # print(
            #     f">>>> memory={memory.size()} src_mask={src_mask.size()} ys={ys.size()} ys_mask={ys_mask.size()}"
            # )

            # @out is size (b x cur_len x d_model), where cur_len is the current
            # len of the sequence as it is extended one at a time. Note it is
            # the same size as ys, since we haven't predicted the next word
            # until below.
            out = model.decode(memory, src_mask, beam_item.ys, ys_mask)
            prob = model.generator(out[:, -1])
            log.info(
                f"beam_decode: ys={beam_item.ys.size()} ys_mask={ys_mask.size()} out={out.size()} prob={prob.size()}"
            )
            # Retrieve top beam-length max next-words. By the pigeonhole
            # principle, this ensures total coverage of possible next-best
            # values. You could implement heuristics here to weight k by current
            # term's likelihood. The sorted=False is because the beam is sorted later.
            #
            # probs is dim and next_word_indices is ____.
            probs, next_word_indices = torch.topk(
                prob, k=beam_length, dim=1, sorted=False
            )
            log.info(
                f">>> probs={probs.size()} next_word_indices={next_word_indices.size()}"
            )
            # TODO: loosely, this is where we could combine beam length b and
            # depth-first search depth d, as follows. For each item in the beam,
            # decode twice forward: find the top-k most likely words for the
            # current item, then for each of these, probe again to find the most
            # likely word sequence of length d (2, 3, 4...). This is from
            # structured prediction, by which b and d can be calibrated for
            # efficiency but also heuristic optimality, to help discover
            # higher-likelihood sequences that are hidden a few steps ahead.
            # This is the same as any tree search or value-backup, accumulate a
            # value and store back pointers. Also note this search could be
            # weighted by probs, to provide slightly better performance.

            # Extend the beam with the top-k next-terms and their probabilities.
            #
            # TODO: I'm foggy on log-softmax, which is the output of the
            # generator. The probs are negative values, and the largest (nearest
            # zero) is the highest prob. Likewise, log-prob addition represents
            # multiplication back in non-log space, hence adding these
            # cumulatively represents the total sequence prob, which is a
            # product.

            log.info(f"LEN: {len(list(zip(probs, next_word_indices)))}")

            for prob, next_word_index in zip(probs, next_word_indices):
                log.info(
                    f">>> next_word_index size: {next_word_index.size()}  prob size: {prob.size()}"
                )
                next_word = next_word_index.data[0]
                next_ys = torch.cat(
                    [
                        beam_item.ys,
                        torch.zeros(batch_size, 1).type_as(src.data).fill_(next_word),
                    ],
                    dim=1,
                )
                log.info(f"Next ys: {next_ys.size()}")

                # TODO: do not append, or rather do not reprocess/decode when
                # next word is the EOS symbol, terminating the sequence. There
                # are multiple questions here such as if terminated sequences
                # should remain in the beam, or search should continue without
                # them, while keeping them separately.
                next_beam.append(BeamItem(ys=next_ys, prob=prob + beam_item.prob))

        # TODO: per other TODOs, optimize beam/next_beam redundancy, and use a heap/pque.
        beam = next_beam
        log.info(f"beam: {beam[0].prob.size()}")
        log.info(
            "Exiting here so you complete beam_decode: make this func consistent with training time "
            + "model input/output sizes. View 'out' and 'generator' and confirm "
            + "usage and sizes are correct."
        )
        exit()
        beam = sorted(beam, key=lambda beam_item: beam_item.prob)
        beam = beam[:beam_length]

    # TODO: return whole beam and print all sequences.
    return beam[0].ys


def hybrid_beam_dfs_decode(
    model: EncoderDecoderModel,
    src,
    src_mask,
    max_len,
    start_symbol,
    beam_length,
    max_depth,
) -> List[torch.Tensor]:
    """
    NOTE: this is interesting but may not be worth the time, as bidirectionality
    probably achieves similar effects.

    An improvement on beam-search during inference is to probe up to some depth
    d using DFS from the current beam. Given a beam length of k and a depth
    parameter d, (1) generate a beam of top-k next words (2) for each node in
    beam, assign its probability as the max sequential probability using greedy
    DFS from that node up to depth d (3) append this max (with its node) to the
    beam.

    This method is equivalent to beam search, except that each node's assigned
    probability is instead a score derived from the max probability when
    extending that node up to depth d. (Realistically, each node is only
    extended heuristically to its top-k successors not it's exponential
    linguistic successors: an incomplete search, but will be good enough.)

    Note that this makes beam-extension a bit ambiguous: for a beam with k
    nodes, do we simply run DFS from each node to get its next score, and
    thereby not extend the current beam to > k nodes (multiple successors)?
    Resolving this entails inspecting the goals and beam and depth search: the
    beam ensure good coverage of successors, while depth-first search ensures
    that we "look-ahead" a sufficient distance to approximately find the best
    sequence without a full exponential search over all possible sequences.

    I think this means there are three parameters: (1) overall beam length k (2)
    child generation length (for each node in beam) k' (3) DFS depth d. Note how
    these separately relate to the real properties of language: the intent of d
    is to ensure that we look-ahead far enough to find high-probability nodes
    that occur later. But the definition of 'probability' depends on the model
    and its biases, as language follows Zip's law such that low-probability
    subject terms (like some local location) may be assigned low probability but
    high contextual probability. Hm...
    """

    # TODO: update me with type info when their structure is understood.
    # Or, remove and replace with tuple if local enough.
    @dataclass
    class BeamItem:
        # The word sequence.
        ys: torch.Tensor
        # The accumulated probability for this sequence.
        prob: float

    # global log
    memory = model.encode(src, src_mask)
    # memory.size()=torch.Size([32, 72, 256]), src=torch.Size([32, 72]) src_mask=torch.Size([32, 1, 72])
    log.info(
        f"memory.size()={memory.size()}, src={src.size()} src_mask={src_mask.size()}"
    )

    # TODO: batchify these ops: intuition says I'm doing this wastefully, but
    # could run inference much faster for a batch of items in the beam at once.
    # Beam search seems extremely amenable to very fast batchification, whereby
    # long beams could be decoded all at once, instead of through iteration.

    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    # The beam is a list of tensors and their associated probabilities.
    # Initialized to the start_symbol, with prob chosen such that subsequent
    # probs accumulate correctly.
    beam: List[BeamItem] = [BeamItem(ys=ys, prob=-1.0)]

    # TODO: revise stopping criteria, i.e. when the whole beam's items achieve
    # some length or are all capped with EOS pads.
    for _ in range(20):
        # For each word in the beam, run the decoder to get its top-k most
        # likely next outputs, appending this to the beam. After predicting for
        # all candidate terms and appending their top-k outputs, sort and
        # truncate the beam back to beam-length. K and beam-length are separate
        # parameters, where beam supports total scope and k is effectively the
        # search radius around candidate words.
        #
        #  TODO: replace beam with a heap to avoid large beam and sorting.
        # TOD: replace deepcopy with a more efficient pattern when code paths harden. This is cope to prevent the infinite loop of appending
        # to the beam while iterating it.
        next_beam = []
        for beam_item in beam:
            # At each time step, mask all subsequent terms to prevent the model looking ahead.
            ys_mask = subsequent_mask(beam_item.ys.size(1)).type_as(src.data)

            # TODO: input to the decoder is given as (1 x seqlen), where b=1.
            # The batch size b could easily be used to support decoding batches
            # of the beam at a time, more efficiently than through iteration.

            # Hybrid beam with greedy depth search:
            # - for each item in the beam
            # - run decoder greedily for d steps, i.e. repeat d times: `next_word = decode(next_word)`
            # - return max_prob, and arg_max sequence for this item to add to beam
            #
            # Other alterations are possible, such as backing up the 'value' of each word per
            # an average over its successors under some strategy, a la Monte Carlo sampling.

            # ys=torch.Size([1, 2]) ys_mask=torch.Size([1, 2, 2])
            # >>>> memory=torch.Size([32, 72, 256]) src_mask=torch.Size([32, 1, 72]) ys=torch.Size([1, 68]) ys_mask=torch.Size([1, 68, 68])
            # print(
            #     f">>>> memory={memory.size()} src_mask={src_mask.size()} ys={ys.size()} ys_mask={ys_mask.size()}"
            # )

            # @out is size (b x cur_len x d_model), where cur_len is the current
            # len of the sequence as it is extended one at a time. Note it is
            # the same size as ys, since we haven't predicted the next word
            # until below.
            out = model.decode(memory, src_mask, beam_item.ys, ys_mask)
            log.info(
                f"ys={beam_item.ys.size()} ys_mask={ys_mask.size()} out={out.size()}"
            )
            prob = model.generator(out[:, -1])
            # Retrieve top beam-length max next-words. By the pigeonhole
            # principle, this ensures total coverage of possible next-best
            # values. You could implement heuristics here to weight k by current
            # term's likelihood. The sorted=False is because the beam is sorted later.
            #
            # probs is dim and next_word_indices is ____.
            probs, next_word_indices = torch.topk(
                prob, k=beam_length, dim=1, sorted=False
            )
            log.info(
                f">>> probs={probs.size()} next_word_indices={next_word_indices.size()}"
            )
            # TODO: loosely, this is where we could combine beam length b and
            # depth-first search depth d, as follows. For each item in the beam,
            # decode twice forward: find the top-k most likely words for the
            # current item, then for each of these, probe again to find the most
            # likely word sequence of length d (2, 3, 4...). This is from
            # structured prediction, by which b and d can be calibrated for
            # efficiency but also heuristic optimality, to help discover
            # higher-likelihood sequences that are hidden a few steps ahead.
            # This is the same as any tree search or value-backup, accumulate a
            # value and store back pointers. Also note this search could be
            # weighted by probs, to provide slightly better performance.

            # Extend the beam with the top-k next-terms and their probabilities.
            #
            # TODO: I'm foggy on log-softmax, which is the output of the
            # generator. The probs are negative values, and the largest (nearest
            # zero) is the highest prob. Likewise, log-prob addition represents
            # multiplication back in non-log space, hence adding these
            # cumulatively represents the total sequence prob, which is a
            # product.

            log.info(f"LEN: {len(list(zip(probs, next_word_indices)))}")

            for prob, next_word_index in zip(probs, next_word_indices):
                log.info(f">>> next_word_index size: {next_word_index.size()}")
                next_word = next_word_index.data[0]
                next_ys = torch.cat(
                    [
                        beam_item.ys,
                        torch.zeros(1, 1).type_as(src.data).fill_(next_word),
                    ],
                    dim=1,
                )
                log.info(f"Next ys: {next_ys.size()}")

                # TODO: do not append, or rather do not reprocess/decode when
                # next word is the EOS symbol, terminating the sequence. There
                # are multiple questions here such as if terminated sequences
                # should remain in the beam, or search should continue without
                # them, while keeping them separately.
                next_beam.append(BeamItem(ys=next_ys, prob=prob + beam_item.prob))

        # TODO: per other TODOs, optimize beam/next_beam redundancy, and use a heap/pque.
        beam = next_beam
        beam = sorted(beam, key=lambda beam_item: beam_item.prob)
        beam = beam[:beam_length]

    # TODO: return whole beam and print all sequences.
    return beam[0].ys

    # # Initially populate the beam using the original code for one step.

    # # For each item in the beam, extend by one step (d=1) and update
    # # probabilities in the beam per the new max values. Process all items in the
    # # beam before updating the beam itself: continually append the new
    # # probabilities, then sort the beam and truncate to beam length again.
    # # Then, repeat the process.

    # # Initially ys is empty except for the start symbol. With the encoder loaded
    # # with input, we can then predict one token at a time, concatenating each
    # # subsequent token onto ys and repeating.
    # ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    # for _ in range(max_len - 1):
    #     # At each time step, mask all subsequent terms to prevent the model looking ahead.
    #     ys_mask = subsequent_mask(ys.size(1)).type_as(src.data)
    #     # ys=torch.Size([1, 2]) ys_mask=torch.Size([1, 2, 2])
    #     # TODO: this is where to implement beam search. Rather than only look
    #     # one word ahead, probe using beam search for subsequences of max value.
    #     out = model.decode(memory, src_mask, ys, ys_mask)
    #     logger.warning(f"ys={ys.size()} ys_mask={ys_mask.size()} out={out.size()}")
    #     prob = model.generator(out[:, -1])
    #     _, next_word = torch.max(prob, dim=1)

    #     # Get the top-k most probable next words from the current output.

    #     # For all of the top-k most probably outputs, run decoding again

    #     logger.info(f">>> next_word size: {next_word.size()}")
    #     next_word = next_word.data[0]
    #     ys = torch.cat(
    #         [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
    #     )
    #     logger.info(f"Next ys: {ys.size()}")

    # return ys


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_encdec_model(V, V, N=2)

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


def yield_tokens(
    data_iter: TGenerator[Tuple[str, str], None, None], tokenizer, index: int
):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# NOTE: unused func, leaving for data io reference. This is from the original
# paper implementation.
def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    log.info("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    log.info("Building English Vocabulary ...")
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
    min_frequency: int = 2,
) -> Vocab:
    """Returns a Vocab object for mapping tokens to indices, built from the passed
    train and validation iters. NOTE: this is for in-language training only,
    whereby the src and tgt sequences are the same, not translation. This
    returns a vocab based on only the src sequence tokens, which assumes that
    the tgt is identical.

    Args:
        * train_iter: An iterator producing (str,str) tups, where the first item
          is the input sentence and the second is the target output.
        * val_iter: The validation iterator, the same type as @train_iter.
        * spacy_en: The spacy english model.
        * min_frequency: The minimum term frequency for a token (word) to be
          included in the vocab. 2 is ideal to drop non-sensical words, but
          requires sufficient data to see most words twice.
    """

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    log.info("Building English Vocabulary ...")
    vocab = build_vocab_from_iterator(
        yield_tokens(itertools.chain(train_iter, val_iter), tokenize_en, index=0),
        min_freq=min_frequency,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab.set_default_index(vocab["<unk>"])

    return vocab


# FUTURE: delete me. Leaving for reference as this vocabulary-building was part
# of the original paper code.
#
# def load_vocab(spacy_de, spacy_en):
#     if not Path("vocab.pt").exists():
#         vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
#         torch.save((vocab_src, vocab_tgt), "vocab.pt")
#     else:
#         vocab_src, vocab_tgt = torch.load("vocab.pt")
#     log.info(f"Finished.\nVocabulary sizes: src={len(vocab_src)} tgt={len(vocab_tgt)}")
#
#     return vocab_src, vocab_tgt


def load_vocab(vocab_path: Path) -> Vocab:
    """load_vocab loads a vocabulary from a pth file."""
    return torch.load(vocab_path)


def save_vocab(vocab: Vocab, vocab_path: Path):
    """Persist all model info. The vocab is a firstclass piece of the model
    since it provides the mapping from vocab to integers for the embedding
    layers.
    """
    torch.save(vocab, vocab_path)


def collate_batch(
    batch,
    src_tokenizer: Callable[[str], List[str]],
    tgt_tokenizer: Callable[[str], List[str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str,
    bs_id: int,
    eos_id: int,
    pad_id: int,
    max_padding: int = 128,
):
    """Batching matters a ton for speed. We want to have very evenly divided
    batches, with absolutely minimal padding. To do this we have to hack a bit
    around the default torchtext batching. This code patches their default
    batching to make sure we search over enough sentences to find tight batches.
    """

    def to_padded_tensor(token_ids: List[int], tokens: List[str]) -> torch.Tensor:
        """to_padded_tensor takes a sequence of token ids, prepending a bos,
        appending the eos id, and then padding the remainder with the padding
        id."""

        # Plus 2 to account for bs and eos id added later.
        if (len(token_ids) + 2) > max_padding:
            log.warning(
                "processed len %d > max_padding of %d (includes bs and eos tokens) in collate_batch for seq %s, truncating training sequence",
                len(token_ids),
                max_padding,
                tokens,
            )
            token_ids = token_ids[: (max_padding - 2)]
        # Minus 2 to account for the bs and eos tokens.
        num_pads = (max_padding - 2) - len(token_ids)

        # Create the initial sentence tensor consisting of the token ids,
        # prepended by a begin-token and appended by an end-token.
        processed = torch.cat(
            [
                bs,
                torch.tensor(
                    token_ids,
                    dtype=torch.int64,
                    device=device,
                ),
                eos,
            ],
            0,
        )

        # Append padding ids to the remaining length of the tensor.
        return pad(
            processed,
            (
                0,
                num_pads,
            ),
            mode="constant",
            value=pad_id,
        )

    bs = torch.tensor([bs_id], device=device)  # <s> token id
    eos = torch.tensor([eos_id], device=device)  # </s> token id

    # FUTURE: iteratorify this code, or at least the caller, such that the
    # entire dataset is not read into memory.
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        src_tokens = src_tokenizer(_src)
        src_ids = src_vocab(src_tokens)
        processed_src = to_padded_tensor(src_ids, src_tokens)
        src_list.append(processed_src)

        tgt_tokens = tgt_tokenizer(_tgt)
        tgt_ids = tgt_vocab(tgt_tokens)
        processed_tgt = to_padded_tensor(tgt_ids, tgt_tokens)
        tgt_list.append(processed_tgt)

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
            bs_id=1,
            eos_id=2,
            pad_id=vocab_src.get_stoi()["<blank>"],
            max_padding=max_padding,
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


def read_line_sentences(fpath: str) -> TGenerator[str, None, None]:
    """read_novel_sentences reads a novel-like file, which is any file
    whose lines extend sentences over one or more lines, like the hucklberry
    finn opensource novel.

    Note that this is super inefficient, because in order to cleanly break on
    lines, I concatenate all lines into a single string to break it on
    delimiters. Hence this is not yet a useful per-line protocol and assumes a
    modest file size.
    """

    def read_lines() -> TGenerator[str, None, None]:
        with open(fpath, "r", encoding="utf8") as lines_file:
            for line in lines_file.readlines():
                yield line.strip()

    def clean_sentences(
        lines: TGenerator[str, None, None],
    ) -> TGenerator[str, None, None]:
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

        for line in lines:
            yield line.lower().translate(t_table)

    return clean_sentences(read_lines())


def read_novel_sentences(fpath: str) -> TGenerator[str, None, None]:
    """
    OBSOLETE: this was built specifically for Gutenberg's Huck Finn.

    read_novel_sentences reads a novel-like file, which is any file
    whose lines extend sentences over one or more lines, like the hucklberry
    finn opensource novel.

    Note that this is super inefficient, because in order to cleanly break on
    lines, I concatenate all lines into a single string to break it on
    delimiters. Hence this is not yet a useful per-line protocol and assumes a
    modest file size.
    """

    def read_lines() -> TGenerator[str, None, None]:
        with open(fpath, "r", encoding="utf8") as novel_file:
            for line in (line.strip() for line in novel_file.readlines()):
                if line and not line.startswith("CHAPTER"):
                    yield line

    def as_sentences(lines: TGenerator[str, None, None]) -> TGenerator[str, None, None]:
        # Common delimiters in huckfinn
        sentence_delimiters = set(["?", ".", ":", "!", ";"])
        # Wastefully join the entire sequence so it can be split on sentence
        # delimiters more simply.
        unsafe_huge_string = " ".join(lines)
        del lines

        start = 0
        end_index = 1
        while start < len(unsafe_huge_string) and end_index < len(unsafe_huge_string):
            # Consume input until we find the start of next sentence.
            if unsafe_huge_string[end_index] in sentence_delimiters:
                sentence = unsafe_huge_string[start:end_index].strip()
                start = end_index + 1
                end_index = start + 1
                if sentence:
                    yield sentence
            end_index += 1

        del unsafe_huge_string

    def clean_sentences(
        lines: TGenerator[str, None, None],
    ) -> TGenerator[str, None, None]:
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

        for line in lines:
            yield line.lower().translate(t_table)

    return clean_sentences(as_sentences(read_lines()))


def get_line_iters(
    fpath: str,
    split: float = 0.8,
    randomize: bool = True,
) -> Tuple[
    TGenerator[Tuple[str, str], None, None], TGenerator[Tuple[str, str], None, None]
]:
    """get_novel_sentence_iters returns a training and validation iterator over the
    lines found in the passed fpath. Each line is treated as an example. Each
    iterator iterates the lines as (line,line) pairs, where the first entry is
    the input and second entry is the target. The lines are shuffled before
    creating their iterators and will be randomly ordered. From fpath, read all
    lines, splits on any of {!?.:}, preserving these at the end of sequences,
    and yields each sentence in this manner as a tuple pair. The return type
    just matches previous requirements, which were for translation tasks where
    tuple (str,tgt) pairs were each in separate languages.

    The lines are shuffled then split into train and validation/test.

    This returns the sentences as-is, no additional parsing. The tokenizer will
    take care of that, plus any other hooks to condition the text.
    """
    lines = list(read_line_sentences(fpath))

    with open("train_lines.txt", "w+", encoding="utf8") as ofile:
        ofile.writelines("\n".join(lines))

    if randomize:
        random.shuffle(lines)

    splitIndex = int(len(lines) * split)
    train_lines = lines[0:splitIndex]
    val_lines = lines[splitIndex:]

    if not train_lines or not val_lines:
        raise ValueError(
            f"Training lines ({len(train_lines)}) or val lines ({len(val_lines)}) empty for fpath={fpath}"
        )

    return ((line, line) for line in train_lines), ((line, line) for line in val_lines)


def get_novel_sentence_iters(
    fpath: str,
    split: float = 0.8,
    randomize: bool = True,
) -> Tuple[
    TGenerator[Tuple[str, str], None, None], TGenerator[Tuple[str, str], None, None]
]:
    """
    OBSOLETE: remove this once completely on the per-line training input.

    get_novel_sentence_iters returns a training and validation iterator over the
    lines found in the passed fpath. Each line is treated as an example. Each
    iterator iterates the lines as (line,line) pairs, where the first entry is
    the input and second entry is the target. The lines are shuffled before
    creating their iterators and will be randomly ordered. From fpath, read all
    lines, splits on any of {!?.:}, preserving these at the end of sequences,
    and yields each sentence in this manner as a tuple pair. The return type
    just matches previous requirements, which were for translation tasks where
    tuple (str,tgt) pairs were each in separate languages.

    The lines are shuffled then split into train and validation/test.

    This returns the sentences as-is, no additional parsing. The tokenizer will
    take care of that, plus any other hooks to condition the text.
    """
    lines = list(read_novel_sentences(fpath))
    with open("train_lines.txt", "w+") as ofile:
        for line in lines:
            ofile.write(line + "\n")

    if randomize:
        random.shuffle(lines)

    splitIndex = int(len(lines) * split)
    train_lines = lines[0:splitIndex]
    val_lines = lines[splitIndex:]

    return ((line, line) for line in train_lines), ((line, line) for line in val_lines)


def create_seq_dataloaders(
    input_lines_path: str,
    device: str,
    vocab_src: Vocab,
    spacy_en: Language,
    batch_size: int = 12000,
    max_padding: int = 128,
    is_distributed: bool = True,
    randomize: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """create_seq_dataloaders returns two date iterators for training and
    validation, for which the training output is the same as the input, for
    in-language prediction. Note that this returns src/tgt lines that are
    index-aligned; Batch then takes offsets tgt by one position.

    @input_lines_path: The path to a line-based training input. This function
    chops the novel into sentences as training data, where the input sentence is
    both the source and the target.

    Objective: ensure that this function creates dataloaders the same as the
    create_dataloaders method, and can load src/tgt examples for a single
    language and text.

    Creates and return sequential dataloaders, for which the source and target
    sequences are identical, i.e. for prediction tasks.
    """

    # Returns tokenized text. Example:
    # - in:  [token.text for token in  spacy_en.tokenizer("foo Bar")]
    # - out: ['foo', 'Bar']
    def tokenize_en(text: str) -> List[str]:
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def collate_fn(batch):
        """Satisfies the torch collate_fn_t interface: 'merges a list of samples
        to form a mini-batch of Tensor(s).  Used when using batched loading from
        a map-style dataset.'"""
        return collate_batch(
            batch,
            tokenize_en,
            tokenize_en,
            vocab_src,
            vocab_src,
            device,
            bs_id=vocab_src.get_stoi()["<s>"],
            eos_id=vocab_src.get_stoi()["</s>"],
            pad_id=vocab_src.get_stoi()["<blank>"],
            max_padding=max_padding,
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
    train_iter, valid_iter = get_line_iters(input_lines_path, randomize=randomize)

    train_iter_map = to_map_style_dataset(train_iter)
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    # Data loader combines a dataset and a sampler, and provides an iterable
    # over the given dataset. There is good DI here, this abstraction wraps all
    # batching, sampling, and conversion from vocab to embedding vectors.
    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None and randomize),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None and randomize),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader


def create_huckfinn_dataloaders(
    novel_path: str,
    device: str,
    vocab_src: Vocab,
    spacy_en: Language,
    batch_size: int = 12000,
    max_padding: int = 128,
    is_distributed: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    OBSOLETE: this was for the Huck Finn loaders. Delete this once completely on per-line training.

    create_seq_dataloaders returns two date iterators for training and
    validation, for which the training output is the same as the input, in other
    words in-language prediction.

    @novel_path: The path to a novel, such as the Gutenberg huckfinn utf8 novel,
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
    def tokenize_en(text: str) -> List[str]:
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def collate_fn(batch):
        """Satisfies the torch collate_fn_t interface: 'merges a list of samples
        to form a mini-batch of Tensor(s).  Used when using batched loading from
        a map-style dataset.'"""
        return collate_batch(
            batch,
            tokenize_en,
            tokenize_en,
            vocab_src,
            vocab_src,
            device,
            bs_id=1,
            eos_id=2,
            pad_id=vocab_src.get_stoi()["<blank>"],
            max_padding=max_padding,
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


class EpochMetrics(BaseModel):
    """
    EpochMetrics is a small serializable class for reporting back per-epoch
    metrics: training and validation loss. This is primarily for visualization
    and inspection of performance.
    """

    epoch: int
    training_loss: float
    validation_loss: float


def train_worker(
    vocab: Vocab,
    spacy_en: Language,
    config: TransformerConfig,
    report_epoch: Callable[[EpochMetrics, EncoderDecoderModel], None] = lambda _: None,
) -> EncoderDecoderModel:
    """train_worker was adapted for training on straightforward language
    prediction: given sequences, encode/decode them in training, and at
    generation time generate one token at a time. Training and validation loss
    are saved per epoch and reported via report_epoch (if passed). No gpu was
    used here, because my laptop doesn't have one and this is just to develop
    the code.

    Returns: the trained EncoderDecoder.
    """
    pad_idx = vocab["<blank>"]
    d_model = config.d_model
    num_layers = config.num_layers
    num_epochs = config.num_epochs

    log.info(
        f"Training params: pad_idx={pad_idx} vocab-len={len(vocab)}\nconfig={config.model_dump_json(indent=2)}"
    )

    model = make_encdec_model(
        len(vocab),
        len(vocab),
        N=num_layers,
        d_model=d_model,
        h=config.h,
        dropout=config.dropout,
    )
    is_main_process = True

    # LabelSmoothing provides regularization. See and run
    # example_label_smoothing.
    criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=0.1)

    train_dataloader, valid_dataloader = create_seq_dataloaders(
        config.data_path,
        config.device,
        vocab,
        spacy_en,
        batch_size=config.batch_size,
        max_padding=config.max_padding,
        is_distributed=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    # Sets the learning rate of each parameter group to the initial lr times a
    # given function. When last_epoch=-1, sets initial lr as lr.
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config.warmup),
    )
    train_state = TrainState()

    for epoch in range(num_epochs):
        model.train()
        log.info(f"CPU Epoch {epoch} Training ====")
        train_loss, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config.accum_iter,
            train_state=train_state,
        )
        log.info("Epoch %d training loss: %f", epoch, train_loss)
        torch.cuda.empty_cache()

        log.info(f"Epoch {epoch} Validation ====")
        model.eval()
        validation_loss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        log.info("Epoch %d validation loss: %f", epoch, validation_loss)

        report_epoch(
            EpochMetrics(
                epoch=epoch, training_loss=train_loss, validation_loss=validation_loss
            ),
            model,
        )

    if is_main_process:
        torch.save(model.state_dict(), f"{config.file_prefix}_final.pt")

    print(f"{num_epochs} epochs completed")

    return model


def load_trained_model(
    src_vocab_size: int, tgt_vocab_size: int, config: TransformerConfig
):
    """load_trained_model creates a model and loads its trained weights from file
    using load_state_dict. IMPORTANT: if the model is being loaded in order to
    perform inference or other production time tasks, ensure that you call
    model.eval() and 'with torch.no_grad()' to disable training-time Dropout and
    detach gradients.
    """
    if not Path(config.model_path).exists():
        raise Exception(f"Model path not found: {config.model_path}")

    model = make_encdec_model(
        src_vocab_size,
        tgt_vocab_size,
        N=config.num_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.h,
        dropout=config.dropout,
    )
    model.load_state_dict(torch.load(config.model_path))

    log.info(
        "Model loaded from %s. Make sure you call model.eval() and 'with "
        + "torch.no_grad()'.",
        config.model_path,
    )

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
    beam_len=1,
):
    results = [()] * n_examples
    for idx in range(n_examples):
        log.info("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        log.info(
            f">> rb.src={rb.src} rb.src.size()={rb.src.size()} rb.src_mask.size()={rb.src_mask.size()}"
        )

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

        log.info(
            "Source Text (Input)        : " + " ".join(src_tokens).replace("\n", "")
        )
        log.info(
            "Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", "")
        )
        # ys = greedy_decode(
        #     model,
        #     rb.src,
        #     rb.src_mask,
        #     max_len,
        #     vocab_src["<s>"],
        # )
        ys = beam_decode(
            model,
            rb.src,
            rb.src_mask,
            max_len,
            vocab_src["<s>"],
            beam_length=beam_len,
            pad_id=pad_idx,
        )

        model_out = ys[0]
        log.info(f"ys={ys.size()} model_out={model_out.size()}")
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))
        print("Model Output               : " + model_txt.replace("\n", "") + "\n")

        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)

    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    log.info("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    log.info("Loading Trained Model ...")

    model = make_encdec_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    log.info("Checking Model Outputs:")
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
