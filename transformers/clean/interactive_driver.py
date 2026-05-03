# Description: given a trained model saved to a pt file, reload it and take
# input from the user interactively for the encoder and decoder.
#
# Note this is the least likely code to remain up to date with changes to the
# transformer... which I kind of accept. This code will easily fall behind.

import os
from pathlib import Path
import logging
import argparse

import torch

import architecture
from transformer_config import TransformerConfig

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
log = logging.getLogger()


def main():
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
    log.info(
        f"""########################################################################
Beginning check-outputs with {args.config} config:
{config.model_dump_json(indent=4)}
########################################################################
"""
    )

    _, spacy_en = architecture.load_tokenizers()

    vocab_path = Path(f"{config.file_prefix}.pth")
    if not vocab_path.exists():
        raise ValueError(
            f"Vocab path {vocab_path} not found; inferred .pth file"
            + f" from config file_prefix: {config.file_prefix}"
        )
    vocab = architecture.load_vocab(vocab_path)

    # device = torch.device(config.device)
    # training_dataloader, validation_dataloader = architecture.create_seq_dataloaders(
    #     config.data_path,
    #     device,
    #     vocab,
    #     spacy_en,
    #     batch_size=config.batch_size,
    #     max_padding=config.max_padding,
    #     is_distributed=False,
    #     is_development=config.is_development(),
    # )

    # Src and tgt vocab length are the same because I've been training a
    # Transformer on its own input sequences as output, for single language
    # prediction, not for translation.
    trained_model = architecture.load_trained_model(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        config=config,
    )
    log.info(f"Model loaded from {config.model_path}")

    # Configure the model for prod-time / non-training mode.
    trained_model.eval()

    with torch.no_grad():
        architecture.interact(
            trained_model, vocab, spacy_en, pad_idx=vocab["<blank>"], eos_string="</s>"
        )


if __name__ == "__main__":
    main()
