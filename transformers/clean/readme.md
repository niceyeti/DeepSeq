# Transformer

This project was derived from the [Annotated Transformer jupyter
notebook](https://nlp.seas.harvard.edu/annotated-transformer/) and [github
repo](https://github.com/harvardnlp/annotated-transformer). The original code
was not for production use, and I extended and fixed the code for my own
training in transformer models. Transformers advanced NLP methods just as I was
leaving graduate school, and this was my opportunity to learn, relying heavily
on the implementation of the original authors.

This repo is primarily for research or demonstration purposes, to learn and
introspect transformer models and their many modifications. I added a beam
search example and certain other advancements to the original model. Please fork
and extend this code as needed. Many other production-grade transformer models
are now in circulation, and I suggest that they be used for production/work
projects.

Most usage of this version is currently for in-language seq2seq models: encode
src vectors for language A to output targets also in language A. I did this for
simplicity, only intending to develop single language tasks; however this is
useful for summarization and multi-modal models. For example, the `make train`
recipe trains a model using input english-text facebook headlines on those same
headlines as the target outputs; this is a degenerate usage of the translation
usage of a seq2seq transformer, but was done for development. In the future I
plan to break out this model into translation, in-language, and multi-modal
seq2seq models.

## Usage

### Testing

0) `make unit-tests`: run unit tests, tests that run fast in a standalone
   manner.
0) Train and verify the test model. This model trains on a small set of
  repeating two-symbol sequences like "a b a b a b..." to verify model
  correctness and for development/debugging. This test is the most useful for
  development and experimentation, to confirm model correctness by training on a
  trivial problem, but a problem that encompasses autoregressive properties
  (repeating non-equal symbols) and vicabulary properties. The model is trained
  using identical src and target input, which is single-language training.
    * `make train-test-model`: trains the test model. This should only take a
      a few minutes on a cpu, training a model on a set of small repeating
      sequences. In more detail, this intentionally overfits on a small dataset
      for which "a b a b a..." is src and also the target. This is extremely
      useful for testing/development that requires a trained model, since the
      data is autoregressive but also very small for short training times. The
      training examples are intentionally constructed such that the vocabulary
      of each sequence is independent of other sequences, to confirm the model
      can learn independent symbols (a trivial case). This could be extended to
      make sequence vocabularies overlap, or to have different autoregressive
      properties, to confirm the model learns these. This is useful to check if
      I've broken anything :) .
    * `make test-model`: runs inference on the test model, using beam search.
      The trained model should overfit and exactly output the input sequences,
      proving that the model is working (albeit, overfitting on such a small
      training set). ![test-model-output](./media/test_model_output.png)

1) `make train`: Train a seq2seq encoder-decoder transformer model on the data
  and model parameters pointed to by the config. Configure a config model to
  point at your target dataset and this will train the config's model.
2) `BEAM_LENGTH=4 make check-outputs`: given a trained model and a config
  pointing to it, this loads it and runs inference by running beam-search,
  predicting sequence outputs using the training data to create output.
  Qualitatively, this is to gauge how closely the model outputs correctly for
  the same training input. ![example trained output](./media/overfit_train.png)


## Known Issues

* this repo is locked to an older version of torch due to the original
  dependence on torchtext. To update it, I need to eliminate torchtext usage and
  then update torch version.

# Personal TODO

TODO: I'm explicitly targeting presentation-ness, whose requirements are:
* api/deployability
* encoder-only and decoder-only architectures

TODO: top-levels I would like to support:

0) Make a data-processing module and put all of the text processing in it
1) `-in` and `-out` command line parameters to point to preprocessed line-based
   files for translation from in -> out
2) DONE: if `-in` only, then use `-in` for both input and output
3) Partially complete: add an encoder-only and decoder-only architecture to be
   used only with `-in`: `-in input.txt --encoder-only` and `--decoder-only`
4) tests for each of these; small trained models will suffice
5) resumable training (this is much further ahead)
6) if needed, for translation it might be nice to devise a small language
  dictionary whose tokens map 1:1 to another set of symbols, i.e. "aaa" ->
  "bbb", "ccc" -> "ddd". Additional semantic structure for pseudo "POS"
  mechanics could be added using a markov model or simple parse tree, depending
  on the test goal: (1) prove transformer learns simple dictionary (2) get it to
  learn more semantic structure (3) by whatever means show how these structures
  are learned. Note that dictionary learning of this kind if essentially
  identical to training a transformer on its own input as output for some small
  language set, even a few simple phrases perhaps.
7) information-retrieval based training metrics: consider including
   task-oriented metrics of enoding by checking how highly the network ranks a
   target phrase as a vector: encode input into context; compare the encoding's
   vector with every other encoded input in the training/validation dataset;
   plot the increase in ranking as training occurs. Recall this was used with
   words for word2vec training.

```bash
# Symmetrical language learning
python training_driver.py -c $(CONFIG_PATH) -in ./data/german_lines.txt -out ./data/english_lines.txt
# Reflexive language learning: provide only '-in'
python training_driver.py -c $(CONFIG_PATH) -in ./data/english_lines.txt
# Encoder-only in-language training
python training_driver.py -c $(CONFIG_PATH) -in ./data/english_lines.txt --enc-only
```
