# Objective: to learn attention/transformer models, I want to train on
# Huckleberry Finn. A lot of this is just retreading old ML/NLP projects and
# knowledge and to begin catching up to progress since the Attention is All
# You Need paper. The original transformer was used for translation, mapping
# english training data to german translations. The assumption here is that
# the same data can be used for input and output in order to train a
# prediction model for prediction/generation tasks instead of translation.

import architecture
import os
import json

# from IPython.display import display

# smoothing_chart = architecture.example_label_smoothing()
# smoothing_chart.save("smoothing.html") display(smoothing_chart)

# train_iter, val_iter = architecture.get_novel_sentence_iters(
#     "./data/huckfinn_utf8.txt") for tup in train_iter(): print(tup)

# TODO: make config a class

# Prod config: this is when you want to actually run training and care about
# saved outputs.
config = dict()
if os.getenv("IS_PROD") == "true":
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "num_layers": 6,  # From the original paper, 6 layers.
        "d_model": 512,  # From the original paper, 512.
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "chuckleberryfinn_model_",
    }
else:
    # Dev config: use this when the saved outputs don't matter and you just want
    # to test pipeline properties end to end: minimum epochs, min cpu, etc.
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 1,
        "accum_iter": 10,
        "num_layers": 2,  # From the original paper, 6 layers.
        "d_model": 100,  # From the original paper, 512.
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "chuckleberryfinn_model_",
        "file_prefix": "deletable",
    }

print(
    f"""########################################################################
Beginning training with config params:\n{json.dumps(config, indent="  ")}
########################################################################
"""
)

_, spacy_en = architecture.load_tokenizers()

# TODO: add a max sequence length parameter. The model has a fixed max input
# size of 512 tokens, and sentences need to be truncated to that length or
# omitted if too long.
train_iter, val_iter = architecture.get_novel_sentence_iters(
    "./data/huckfinn_utf8.txt")

vocab = architecture.build_en_vocabulary(train_iter, val_iter, spacy_en)

architecture.my_train_worker(vocab, spacy_en, config)

# TODO: 8/31: left off here. The task is to complete the my_generation method to generate
# sequences using greedy decoding or possibly other strategies. The ground level task
# is to rewrite the DataLoaders to convert text to input for the model.
# 1) input a single word and predict the next
# 2) (hopefully) encode an entire sequence and then try to predict the output one at a time (like translation)
# 'check_outputs' looks promising and should be a good starting point.
