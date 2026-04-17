"""Description: this module is for code models, in the software engineering
sense not ML models. These models should have few/no non-builtin
dependencies."""

from pydantic import BaseModel


class EpochMetrics(BaseModel):
    """EpochMetrics is a small serializable class for reporting back per-epoch
    metrics: training and validation loss. This is primarily for visualization
    and inspection of performance.
    """

    epoch: int
    training_loss: float
    validation_loss: float
    train_duration: float
    validation_duration: float
    dt_8601: str


class TrainState:
    """Track number of steps, examples, and tokens processed."""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
