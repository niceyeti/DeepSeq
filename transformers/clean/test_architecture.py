import os
import itertools
from tempfile import NamedTemporaryFile
from typing import List
from unittest import TestCase

import torch

import architecture


class TestArchitecture(TestCase):
    def test_foo(self):
        self.assertTrue(True)

    def test_inference(self):
        ys = architecture.inference_test()
        self.assertTrue(ys is not None, "Inference check passes if no errors raised")

    def test_encoder_inference(self):
        ys = architecture.encoder_inference_test()
        self.assertTrue(ys is not None, "Inference check passes if no errors raised")

    def test_decoder_inference(self):
        ys = architecture.decoder_inference_test()
        self.assertTrue(ys is not None, "Inference check passes if no errors raised")

    def test_seq_data_loaders(self):
        # Note: I left some bad content/characters in this test input
        # intentionally... it seems good to test.
        training_file_content = """israel faces a new military and political challenge in the gaza strip as palestinian activists have pledged weeks of protests after israeli soldiers opened fire friday on at-times violent demonstrators killing at least 15
the new york times was awarded three pulitzer prizes including the award for public-service journalism while the reuters news agency won two
oracle corp 's shares sank 6 5% in late trading monday after the company disappointed wall street with its guidance for cloud-computing revenue in the current quarter--the third-consecutive period it has done so
pyongyang plans to take steps to demonstrate the closure of the punggye ri site to the world south koreas presidential office said adding to momentum for a deal on the regimes nuclear program after last weeks historic talks
kim jong un became the first north korean leader to set foot in the south since the korean war when he stepped across the military demarcation line friday and shook hands with moon jae in the south korean president
nintendo plans to pick up the production pace for its hit switch game console next year showing the companys deepening confidence in the success of the device"""

        training_lines = training_file_content.split("\n")

        with NamedTemporaryFile(
            mode="w+", delete=True, suffix=".txt", encoding="utf8"
        ) as temp_file:
            # Swap the tempfile for a local one when you want to inspect outputs
            if "DEBUG" in os.environ:
                temp_file = open("./foo.txt", "w+", encoding="utf8")

            temp_file.write("\n".join(training_lines))
            temp_file.flush()

            _, spacy_en = architecture.load_tokenizers()
            train_iter, val_iter = architecture.get_line_iters(temp_file.name)
            vocab: architecture.Vocab = architecture.build_en_vocabulary(
                train_iter, val_iter, spacy_en, min_frequency=1
            )

            # Given: a training and validation batch iterator
            max_padding = 128
            train_iter, val_iter = architecture.create_seq_dataloaders(
                temp_file.name,
                "cpu",
                vocab,
                spacy_en,
                batch_size=12000,
                max_padding=max_padding,
                is_distributed=False,
                # Set randomize to false, since this test depends on
                # deterministic ordering to be able to test things.
                randomize=False,
            )

            # Then: the combined iterators have the same number of examples as
            # training file lines. The properties of either are ignored, we
            # are more concerned with top-level testing of their content.

            batches: List[architecture.Batch] = []
            for b in itertools.chain(train_iter, val_iter):
                batches.append(architecture.Batch(b[0], b[1], pad_id=vocab["<blank>"]))

            # Then there are two batches: one train, one validation
            self.assertEqual(
                len(batches),
                2,
                f"Expected 2 examples but got {len(batches)} {train_iter}",
            )

            train_batch: architecture.Batch = batches[0]
            validation_batch: architecture.Batch = batches[1]
            # Validation line untested: testing only the training batch seemed sufficient.
            _ = validation_batch

            expected_train_seqs = 4

            expected_src_size = torch.Size([expected_train_seqs, max_padding])
            expected_tgt_size = torch.Size([expected_train_seqs, max_padding - 1])

            # Then batch shape of src and tgt is
            self.assertEqual(
                train_batch.src.shape,
                expected_src_size,
                f"Expected train batch src shape {expected_src_size} (b x max_padding) but got {train_batch.src.shape}\ntgt={train_batch.tgt[0]}\n\nsrc={train_batch.src[0]}",
            )

            self.assertEqual(
                train_batch.tgt.shape,
                expected_tgt_size,
                f"Expected train batch src shape {expected_tgt_size} (b x max_padding-1) but got {train_batch.tgt.shape}\ntgt={train_batch.tgt[0]}\n\nsrc={train_batch.src[0]}",
            )

            # TODO: fix this fragile maintenance. The first training line in the
            # batch is given by the reversal of the training lines, which for a
            # split of 0.8 is lines 0-3 above, and thus the "pyongyang" line
            # ends up first, hence index 3.
            first_line = training_lines[3]
            first_line_tokens = first_line.split(" ")
            # Then first sequence in batch is the same length as the first token
            # sequence, plus one prepended <s> id and suffixed with one </s> id,
            # and the rest is padding. Index is -1 because the lines get
            # reversed; I don't intend to try and order this such that the first
            # line in the file is the first in the batch because something
            # internal reverses it and I will accept the behavior as such.
            first_batch_sequence = train_batch.src[-1]
            seq_tokens = [
                vocab.lookup_token(token_id) for token_id in first_batch_sequence
            ]
            # Then eos is located one position past the last word.
            eos_index = len(first_line.split(" ")) + 1
            src_token_ids = first_batch_sequence[1:eos_index]
            itos = vocab.get_itos()
            # Decode the batch back to source words and ensure it matches the input.
            src_tokens = [itos[token_id] for token_id in src_token_ids]
            decoded_sentence = " ".join(src_tokens)

            # TODO: need to pass eos and bs ids in a better manner. These ids
            # need to be placed somewhere, not hidden down in the methods.
            self.assertEqual(
                first_batch_sequence[0],
                0,
                f"Expected bs id to equal 0 but got {first_batch_sequence[0]}",
            )

            self.assertEqual(
                first_batch_sequence[eos_index],
                1,
                f"Expected eos id to equal 1 but got {first_batch_sequence[eos_index]}  ({vocab.lookup_token(first_batch_sequence[eos_index])})  neighborhood is: {first_batch_sequence[eos_index-5:min(eos_index+5, first_batch_sequence.size()[0])]}\nand tokens: {seq_tokens} \n for test line: {first_line}",
            )

            self.assertEqual(
                vocab.lookup_token(first_batch_sequence[eos_index]),
                "</s>",
                f"Expected eos token to equal </s> but got {vocab.lookup_token(first_batch_sequence[eos_index])}  with tokens:\n{seq_tokens}",
            )

            self.assertEqual(
                vocab.lookup_token(first_batch_sequence[eos_index + 1]),
                "<blank>",
                f"Expected first padding token to equal <blank> but got {vocab.lookup_token(first_batch_sequence[eos_index + 1 ])}  with tokens:\n{seq_tokens}",
            )

            self.assertEqual(
                decoded_sentence,
                first_line,
                f"Expected line from first batch to equal input line but got:\ndecoded={decoded_sentence}\ninput={first_line}",
            )
