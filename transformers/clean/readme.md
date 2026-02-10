
TODO: I'm explicitly targeting presentation-ness, whose requirements are:
* api/deployability
* using torch
* prediction
* encoder-only and encoder-decoder architectures

1/25/26 Top-level status: I created a simple fb_data training lines file, but it
needs to be preprocessed.


TODO: top-levels I would like to support:
1) `-in` and `-out` command line parameters to point to preprocessed line-based
   files for translation from in -> out
2) DONE: if `-in` only, then use `-in` for both input and output
3) Add an encoder-only architecture to be used only with `-in`
4) tests for each of these; small trained models will suffice
5) resumable training (this is much further ahead)
6) if needed, for translation it might be nice to devise a small language
  dictionary whose tokens map 1:1 to another set of symbols, i.e. "aaa" ->
  "bbb", "ccc" -> "ddd". Additional semantic structure for pseudo "POS"
  mechanics could be added using a markov model or simple parse tree, depending
  on the test goal: (1) prove transformer learns simple dictionary (2) get it to
  learn more semantic structure (3) by whatever means show how these structures
  are learned

```bash
# Symmetrical language learning
python training_driver.py -c $(CONFIG_PATH) -in ./data/german_lines.txt -out ./data/english_lines.txt
# Reflexive language learning: provide only '-in'
python training_driver.py -c $(CONFIG_PATH) -in ./data/english_lines.txt
# Encoder-only in-language training
python training_driver.py -c $(CONFIG_PATH) -in ./data/english_lines.txt --enc-only
```
