# Description: given a trained model saved to a pt file, reload it and show its
# outputs. This is currently messy because it involves re-creating dependencies
# of training, such as the vocab and train/validation iters, etc.

import os
import json
import architecture
import torch


file_prefix = os.getenv("FILE_PREFIX", "")
if not file_prefix:
    raise ValueError(
        "Set FILE_PREFIX to the prefix of the model's files to load, i.e. ."
    )

model_path = os.getenv("MODEL_PATH", "")
if not model_path:
    raise ValueError("Set MODEL_PATH to the trained .pt model to load.")

config = dict()
with open(f"{file_prefix}.json", "r") as ifile:
    config = json.load(ifile)
print("Reloaded config: ", json.dumps(config, indent=True))

_, spacy_en = architecture.load_tokenizers()

# TODO: add a max sequence length parameter. The model has a fixed max input
# size of 512 tokens, and sentences need to be truncated to that length or
# omitted if too long.
train_iter, val_iter = architecture.get_novel_sentence_iters(config["data_path"])

# vocab = architecture.build_en_vocabulary(train_iter, val_iter, spacy_en)
vocab = torch.load(f"{file_prefix}.pth")

train_dataloader, valid_dataloader = architecture.create_seq_dataloaders(
    config["data_path"],
    torch.device("cpu"),
    vocab,
    spacy_en,
    batch_size=config["batch_size"],
    max_padding=config["max_padding"],
    is_distributed=False,
)

trained_model = architecture.my_load_trained_model(
    vocab,
    vocab,
    config,
    model_path,
)

# TODO: write a beam-search method to search for probable sequences.
architecture.check_outputs(
    valid_dataloader,
    trained_model,
    vocab,
    vocab,
    n_examples=7,
    pad_idx=vocab["<blank>"],
    eos_string="</s>",
)
