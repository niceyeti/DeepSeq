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
import json
import argparse
from pathlib import Path

import architecture
from model_config import TransformerConfig
import matplotlib.pyplot as plt


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
log = logging.getLogger()


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

    with open(args.config, "r", encoding="utf8") as config_file:
        config_json = "".join(
            [
                line.strip()
                for line in config_file.readlines()
                if not line.strip().startswith("//")
            ]
        )
        config: TransformerConfig = TransformerConfig.model_validate_json(config_json)
        config = config.read_from_env()

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
    train_iter, val_iter = architecture.get_novel_sentence_iters(config.data_path)

    vocab = architecture.build_en_vocabulary(train_iter, val_iter, spacy_en)
    architecture.save_vocab(vocab, f"{config.file_prefix}.pth")

    loss_path = Path(f"{config.file_prefix}.loss")

    def append_loss(metrics: architecture.EpochMetrics):
        with open(loss_path, "a+", encoding="utf8") as loss_file:
            loss_file.write(metrics.model_dump_json(indent=None) + "\n")

    architecture.my_train_worker(vocab, spacy_en, config, report_epoch=append_loss)

    with open(loss_path, "r", encoding="utf8") as loss_file:
        losses = [
            architecture.EpochMetrics.model_validate_json(line)
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
    plt.savefig(Path(f"{config.file_prefix}.loss.png"))


if __name__ == "__main__":
    main()
