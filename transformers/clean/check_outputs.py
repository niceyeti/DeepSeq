# Description: given a trained model saved to a pt file, reload it and show its
# outputs. This is currently messy because it involves re-creating dependencies
# of training, such as the vocab and train/validation iters, etc.

import os
import json
from pathlib import Path
import logging
import argparse

import torch

import architecture
from model_config import TransformerConfig


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
log = logging.getLogger()


def main():
    """"""

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
    # train_iter, val_iter = architecture.get_novel_sentence_iters(config.data_path)

    vocab = architecture.load(f"{config.file_prefix}.pth")

    train_dataloader, valid_dataloader = architecture.create_seq_dataloaders(
        config.data_path,
        torch.device(config.device),
        vocab,
        spacy_en,
        batch_size=config.batch_size,
        max_padding=config.max_padding,
        is_distributed=False,
    )

    trained_model = architecture.my_load_trained_model(
        vocab,
        vocab,
        config,
        config.model_path,
    )

    architecture.check_outputs(
        valid_dataloader,
        trained_model,
        vocab,
        vocab,
        n_examples=7,
        pad_idx=vocab["<blank>"],
        eos_string="</s>",
    )
