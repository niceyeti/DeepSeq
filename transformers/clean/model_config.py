from __future__ import annotations

import os
import logging

from pydantic import BaseModel


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
log = logging.getLogger()

# Ignore file naming, this module may contain multiple configs in the future,
# i.e. for Encoder-only architectures and other variants.


class TransformerConfig(BaseModel):
    """
    ModelConfig contains all of the parameters for the transformer. This is
    directly serializable to json and should remain as such.

    All fields are overridable via env vars using
    'TRANSFORMER_[uppercase-field-name]=[value].'

    ModelConfig operates in two modes: training and prod. In training, there is
    no previous model or vocabulary stored, and the config definition drives the
    creation and some peristence info for the model. In production, a model and
    vocabulary already exist and must be specified to be loaded in. For now, the
    idea is that the behavioral differences can be settled by simply splitting
    the drivers for training and inference/prod.
    """

    # The batch size for training
    batch_size: int = 32
    # Whether or not training is distributed
    distributed: bool = False
    # The number of epochs for which to train, as a static value.
    num_epochs: int = 36
    # Number of iterations to accumulate weight updates.
    accum_iter: int = 10
    # The number of layers, 6 in the original paper. The encoder and decoder will
    # each have this number of layers.
    num_layers: int = 6
    # The model dimension, 512 in the original paper.
    d_model: int = 512
    # Dropout rate.
    dropout: float = 0.1
    # The dimenson of the feed-forward network appended to each attention
    # encoder/decoder layer.
    d_ff: int = 2048
    # The number of heads in multi-headed attention components.
    h: int = 8
    # The base learning rate.
    base_lr: float = 1.0
    # Max padding is the maximum padding but also effectively the maximum sequence
    # length on which to train.
    max_padding: int = 72
    warmup: int = 3000
    # The prefix by which models will be persisted and read back in. Model
    # progress is saved at each epoch in a pt file with this prefix.
    file_prefix: str = "./models/test/fb_news"
    # The training data path.
    data_path: str = "./data/fb_lines.txt"
    # The path to a saved model. This is only used in inference/prod to load
    # a saved model.
    #
    # TODO: keep an eye on this attribute, it may be derivable from file_prefix instead.
    model_path: str = "./models/test/fb_news_final.pt"
    # Device must be either "cpu" or "gpu", and is passed directly to torch.
    # See torch docs.
    device: str = "cpu"

    def read_from_env(self) -> TransformerConfig:
        """
        read_from_env can be called to override any config field from env vars,
        prefixed by 'Transformer' and formatted like "TRANSFORMER_[uppercase
        attribute]=[new value]". The string value of the env var is passed to
        the ctor for that field's type, thus inheriting pythonic semantics for
        every field and only support basic types.

        I'm only doing this because I'm unaware of a builtin way to do it using
        pydantic, but this is also the simplest."""
        clone = self.model_copy()
        for key, prev_val in clone.__dict__.items():
            env_var_name = "TRANSFORMER_" + key.upper()
            if env_var_name in os.environ:
                # 'TRANSFORMER_[attribute name]' is set, so map it to its target
                # type and override the current value. We borrow from python's
                # own type rules for converting the string env var value,
                # passing this directly to the constructor for the type.
                new_val = os.environ[env_var_name]
                log.info(
                    "%s overriding config.%s from %s to %s",
                    env_var_name,
                    key,
                    prev_val,
                    new_val,
                )
                clone.__dict__[key] = type(prev_val)(new_val)

        return self.model_validate(clone)
