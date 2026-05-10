# Description: given a trained model saved to a pt file, reload it and show its
# outputs. This is currently messy because it involves re-creating dependencies
# of training, such as the vocab and train/validation iters, etc.

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
    """This driver loads a built model and runs inference to generate and print
    a few sample outputs."""

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

    # TODO: add a max sequence length parameter. The model has a fixed max input
    # size of 512 tokens, and sentences need to be truncated to that length or
    # omitted if too long.
    # train_iter, val_iter = architecture.get_novel_sentence_iters(config.data_path)

    vocab_path = Path(f"{config.file_prefix}.pth")
    if not vocab_path.exists():
        raise ValueError(
            f"Vocab path {vocab_path} not found; inferred .pth file"
            + f" from config file_prefix: {config.file_prefix}"
        )
    vocab = architecture.load_vocab(vocab_path)

    device = torch.device(config.device)
    training_dataloader, validation_dataloader = architecture.create_seq_dataloaders(
        config.data_path,
        device,
        vocab,
        spacy_en,
        batch_size=config.batch_size,
        max_padding=config.max_padding,
        is_distributed=False,
        is_development=config.is_development(),
    )

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

    def top_k_strategy(
        smax_prob: torch.Tensor,
        next_token_index: int,
        softmax_dim: int,
        beam_length: int,
    ):
        """When running beam search, we encode the source completely and then
        generate tgt output one token at a time, each of which is selected from
        the softmax output at @softmax_dim stored in @probs, which has the |V|
        outputs of the model's generator layer (a softmax func). This func is
        called at each timestep, and simply returns the top-k terms, each with
        their non-normalized probability. This is greedyish, as the model will
        usually precisely encode its training data, such that the top-1 term
        contains the overwhelming portion of the softmax cdf.

        The caller takes the list of tuples and uses them to update the beam. It
        extends the beam-item for the current token with the probability of each
        top-k next-word, along with its id, adding the word's prob to the
        beam-items prob. Hence the prob stored in each beam item is the
        accumulated sequential log-probability of the complete sequence
        represented by that beam item. This func returns the model's actual
        log-probability of each next term; however, popular variants normalize
        these outputs, apply smoothing, or other nucleus-like strategies, such
        that the beam is populated and sorted not with formal probabilities but
        with score-like values.
        """

        probs, next_word_indices = torch.topk(
            # Sorted=False, since the entire beam is sorted and truncated by
            # beam_length later, hence there is no need to sort here.
            smax_prob,
            # k=beam_length ensures that, by the pigeonhole principle, all k
            # outputs will be considered in the output.
            k=beam_length,
            dim=softmax_dim,
            sorted=False,
        )
        # The tensors are out=torch.Size([64, 72, 512]) probs=torch.Size([64,
        # 72, 30]) next_word_indices=torch.Size([64, 72, 30]). Accordingly,
        # probs contains the actual probability, and next_word_indices their
        # corresponding indices.
        log.info(f"probs={probs.size()}  next_word_indices={next_word_indices.size()}")

        return [
            (prob.item(), next_word_index.item())
            for (prob, next_word_index) in zip(
                probs[0, next_token_index, :],
                next_word_indices[0, next_token_index, :],
            )
        ]

    def harmonic_smoothing_strategy(
        smax_prob: torch.Tensor,
        next_token_index: int,
        softmax_dim: int,
        beam_length: int,
    ):
        """See top_k_strategy for calling semantics. This is a loose attempt to diversify
        the outputs by replacing each estimated token probability with a pseduo-probability
        defined by its list-rank 1/r, using harmonic smoothing.
        """
        probs, next_word_indices = torch.topk(
            smax_prob,
            k=beam_length,
            dim=softmax_dim,
            sorted=False,
        )

        token_probs = [
            (prob.item(), next_word_index.item())
            for (prob, next_word_index) in zip(
                probs[0, next_token_index, :],
                next_word_indices[0, next_token_index, :],
            )
        ]
        # Normalize probs using harmonic smoothing. Note that a well-fitted
        # model will still inductively predict the top-prob word, since it still
        # has the highest prob at each step, and inductively the encoded
        # sequence will generally be predicted exactly. Could add noise here to
        # ensure the output still have some diversity in its sampling. Example
        # output for a list of 4 tokens: [ (4 - 0 + 1) / (4 + 2), ...  ] = [5/6,
        # 3/6, ]
        N = float(len(token_probs))
        smoothed = [
            (float(N - i) / N, next_word_index)
            for i, (_, next_word_index) in enumerate(token_probs)
        ]
        denom = float(sum(tup[0] for tup in smoothed))
        return [(x / denom, next_word_index) for x, next_word_index in smoothed]

    with torch.no_grad():
        architecture.check_outputs(
            training_dataloader,
            trained_model,
            vocab,
            vocab,
            sample_outputs=harmonic_smoothing_strategy,
            n_examples=7,
            pad_idx=vocab["<blank>"],
            eos_string="</s>",
            beam_len=config.beam_length,
        )


if __name__ == "__main__":
    main()
