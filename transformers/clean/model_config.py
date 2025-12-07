from pydantic import BaseModel

# Ignore file naming, this module may contain multiple configs in the future,
# i.e. for Encoder-only architectures and other variants.


class TransformerConfig(BaseModel):
    """
    ModelConfig contains all of the parameters for the transformer. This is
    directly serializable to json and should remain as such.

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
    file_prefix: str = "chuckleberryfinn_model"
    # The training data path.
    data_path: str = "./data/huckfinn_utf8.txt"
    # The path to a saved model. This is only used in inference/prod to load
    # a saved model.
    model_path: str = "./model_final.pt"
