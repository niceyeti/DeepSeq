# Description: given a trained model saved to a pt file, reload it and show its
# outputs. This is currently messy because it involves re-creating dependencies
# of training, such as the vocab and train/validation iters, etc.

import architecture
import torch


_, spacy_en = architecture.load_tokenizers()

# TODO: add a max sequence length parameter. The model has a fixed max input
# size of 512 tokens, and sentences need to be truncated to that length or
# omitted if too long.
train_iter, val_iter = architecture.get_novel_sentence_iters(
    "./data/huckfinn_utf8.txt")

vocab = architecture.build_en_vocabulary(train_iter, val_iter, spacy_en)

train_dataloader, valid_dataloader = architecture.create_seq_dataloaders(
    # TODO: add path to config
    "./data/huckfinn_utf8.txt",
    torch.device("cpu"),
    vocab,
    spacy_en,
    # TODO: hard-coded to match what training used... this is easily broken.
    batch_size=32,
    max_padding=72,
    is_distributed=False,
)

trained_model = architecture.my_load_trained_model(
    vocab, vocab, "chuckleberryfinn_model_final.pt")

architecture.check_outputs(valid_dataloader, trained_model, vocab, vocab)
