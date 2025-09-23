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
# piece of the model is reproducible/deserializable.


import architecture
import os
import json
import torch

# TODO: make config a class

# Prod config: this is when you want to actually run training and care about
# saved outputs.

# Default dev config to test pipeline properties end to end: minimum epochs, min
# cpu, etc.
config_name = "default"
config = {
    "batch_size": 8,
    "distributed": False,
    "num_epochs": 1,
    # The number of batch iterations to accumulate before calling optimizer.step().
    "accum_iter": 4,
    "num_layers": 2,
    "d_model": 64,
    "d_ff": 2048,
    "h": 8,
    "dropout": 0.1,
    "base_lr": 1.0,
    "max_padding": 72,
    "warmup": 3000,
    "file_prefix": "deletable",
    "data_path": "./data/huckfinn_utf8.txt",
}
config_name = "default"

if os.getenv("IS_PROD") == "true":
    config_name = "prod"
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 36,
        # The number of batch iterations to accumulate before calling optimizer.step().
        "accum_iter": 10,
        "num_layers": 6,  # From the original paper, 6 layers.
        "d_model": 256,  # From the original paper, 512.
        "d_ff": 2048,
        "h": 8,
        "dropout": 0.1,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "chuckleberryfinn_model",
        "data_path": "./data/huckfinn_utf8.txt",
    }


print(
    f"""########################################################################
Beginning training with {config_name} config:
{json.dumps(config, indent="  ")}
########################################################################
"""
)

_, spacy_en = architecture.load_tokenizers()

# TODO: add a max sequence length parameter. The model has a fixed max input
# size of 512 tokens, and sentences need to be truncated to that length or
# omitted if too long.
train_iter, val_iter = architecture.get_novel_sentence_iters(config["data_path"])

vocab = architecture.build_en_vocabulary(train_iter, val_iter, spacy_en)
# Persist all model info. The vocab is a firstclass piece of the model since it
# provides the mapping from vocab to integers for the embedding layers.
file_prefix = config["file_prefix"]
torch.save(vocab, f"{file_prefix}.pth")
with open(f"{file_prefix}.json", "w+") as cfg_file:
    json.dump(config, cfg_file, indent=True)

model = architecture.my_train_worker(vocab, spacy_en, config)
