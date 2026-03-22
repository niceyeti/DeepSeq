# Objective: to learn attention/transformer models, I want to train on
# Huckleberry Finn. A lot of this is just retreading old ML/NLP projects and
# knowledge and to begin catching up to progress since the Attention is All You
# Need paper. The original transformer was used for translation, mapping english
# training data to german translations. The assumption here is that the same
# data can be used for input and output in order to train a prediction model for
# prediction/generation tasks instead of translation.
#
# TODO: the training vocabulary and sequences are currently coupled to the
# training such that generating from the model requires loading in exactly the
# same vocabulary (size) and mappings. Persist the vocabulary and embedding
# lookup tables at train time; in fact, save the entire config such that every
# piece of the model is reproducible/deserializable from training.

import logging
import os
import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

import architecture
from architecture import EpochMetrics, EncoderDecoderModel
from transformer_config import TransformerConfig

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
log = logging.getLogger()


def plot_losses(loss_path: Path, save_path: Path):
    with open(loss_path, "r", encoding="utf8") as loss_file:
        losses = [
            EpochMetrics.model_validate_json(line)
            for line in filter(len, loss_file.readlines())
        ]

    training_losses = [loss.training_loss for loss in losses]
    validation_losses = [loss.validation_loss for loss in losses]
    epochs = [loss.epoch for loss in losses]

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses)
    plt.title("Training Loss Per Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_losses)
    plt.title("Validation Loss Per Epoch")
    plt.tight_layout()
    plt.savefig(save_path)


def main():
    """
    Train supports two workflows:
        1) set CONFIG_PATH to the path to a config for a model to train or load
        2) pass nothing: by default this will train on a default dataset used as
           a smoke test.
    """

    parser = argparse.ArgumentParser(
        prog="Transformer",
        description="This package implements basic transformer components from The Annotated Transformer.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=Path,
        default="./deployment/default_config.jsonc",
        help="The path to the jsonc config. See ./deployment/default_config.jsonc example.",
    )
    args = parser.parse_args()

    config = TransformerConfig.load(args.config).read_from_env()

    # Initialize the parent output directories for model artifacts, some of
    # which are created at train time.
    model_dir = Path(config.file_prefix).parent
    os.makedirs(model_dir, exist_ok=True)

    log.info(
        f"""########################################################################
Beginning training with {args.config} config:
{config.model_dump_json(indent=4)}
########################################################################
"""
    )
    _, spacy_en = architecture.load_tokenizers()

    # TODO: add a max sequence length parameter. The model has a fixed max input
    # size of 512 tokens, and sentences need to be truncated to that length or
    # omitted if too long.
    train_iter, val_iter = architecture.get_line_iters(config.data_path)

    vocab = architecture.build_en_vocabulary(
        train_iter, val_iter, spacy_en, min_frequency=1
    )

    architecture.save_vocab(vocab, f"{config.file_prefix}.pth")

    loss_path = Path(f"{config.file_prefix}.loss")
    # zero the existing loss file
    loss_path.write_text("", encoding="utf8", newline="")

    only_once = os.environ.get("ONCE", "") != ""

    def persist_epoch(metrics: EpochMetrics, model: EncoderDecoderModel):
        """persist_epoch saves the metrics for the epoch to file so that once
        training is completed, they can be read and plotted. It also tracks the
        best validation error and saves the model weights, so this is the
        callback by which to implement persistence logic (MLFlow integration,
        etc).
        """
        # Save model weights if this one has the best validation error. Note
        # this is where the entire model could be saved every n-epochs, etc, but
        # saving only the best model per validation error is all we're
        # interested in now.
        with open(loss_path, "r", encoding="utf8") as loss_file:
            losses = [
                EpochMetrics.model_validate_json(line)
                for line in loss_file.readlines()
                if line.strip()
            ]
        min_validation_error = min(
            map(lambda l: l.validation_loss, losses), default=99999
        )
        if metrics.validation_loss < min_validation_error:
            torch.save(model.state_dict(), f"{config.file_prefix}_best.pt")

        # Append the new metrics. Note this should be done after comparing the
        # losses on file, or the new metrics would be in the compared set.
        with open(loss_path, "a+", encoding="utf8") as loss_file:
            loss_file.write(metrics.model_dump_json(indent=None) + "\n")

        if only_once:
            # For debugging, it is often useful to max the log level and bail
            # after a single iteration, to review tensor size agreement, etc.
            log.info("ONCE passed, exiting after a single iteration.")
            exit(1)

    try:
        architecture.train_worker(vocab, spacy_en, config, report_epoch=persist_epoch)
    finally:
        # Plots losses, even on ctrl+c sigterm.
        plot_losses(loss_path, Path(f"{config.file_prefix}.loss.png"))


if __name__ == "__main__":
    main()
