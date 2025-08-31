import architecture

# from IPython.display import display

# smoothing_chart = architecture.example_label_smoothing()
# smoothing_chart.save("smoothing.html")
# display(smoothing_chart)

# train_iter, val_iter = architecture.get_novel_sentence_iters(
#     "./data/huckfinn_utf8.txt")
# for tup in train_iter():
#     print(tup)

print("training...")

# TODO: make config a class
config = {
    "batch_size": 32,
    "distributed": False,
    "num_epochs": 8,
    "accum_iter": 10,
    "base_lr": 1.0,
    "max_padding": 72,
    "warmup": 3000,
    "file_prefix": "multi30k_model_",
}

_, spacy_en = architecture.load_tokenizers()

train_iter, val_iter = architecture.get_novel_sentence_iters(
    "./data/huckfinn_utf8.txt")

vocab = architecture.build_en_vocabulary(train_iter, val_iter, spacy_en)

architecture.my_train_worker(vocab, spacy_en, config)
