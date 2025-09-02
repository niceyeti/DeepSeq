# Attention Model Self Course

This code is lifted directly from "The Annotated Transformer" code, for
self-directed learning of both attention models and torch.

A primary objective is to implement a predictive attention model, which is an
attention model whose training source/target pairs are identical, rather than
being source/target sequences from different languages. This is a known
degenerate use of attention models for technically obvious reasons: the output
of encoder blocks into decoder blocks is essentially redundantish, since the
decoder receives the same input as the encoder.

Note: if you want to do production transformer development, it is probably
better to use huggingface or sklearn trasnformer models.
