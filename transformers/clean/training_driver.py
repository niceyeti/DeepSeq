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

import architecture
from model_config import TransformerConfig


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

    log.info(
        f"""########################################################################
Beginning training with {args.config} config:
{config.model_dump_json(indent="  ")}
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


if __name__ == "__main__":
    main()
