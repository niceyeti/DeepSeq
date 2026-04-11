# Description: this makefile is intended to mock out similar workflows to the
# desired api of a transformer for train/test.
#
# To develop or run the transformer, the most useful and important recipe for
# demonstration is to run "make train-test-model" followed by "make test-model".
# This workflow trains the transformer on a very simple input dataset of
# repeating symbols to verify model functionality, autoregression, and other
# model properties. This simple example is the best iterative development loop
# for demos and for model modification.

SHELL:=/bin/bash

DATA_PATH?=./data/fb_lines.txt

MODEL_DIR=./trained_models
TEST_MODEL_DIR=$(MODEL_DIR)/test

CONFIG_PATH?=./deployment/default_config.jsonc
# This config points to the simple-sequence training/output workflow for
# development and testing.
TEST_MODEL_CONFIG_PATH?=./deployment/model_verification_test_config.jsonc


# Use DEBUG to view gritty model outputs, tensor dimensions, etc.
LOG_LEVEL?=WARNING

# Note: literature has a lot to say about this, such as that a beam-length of
# 3-6 is optimal. Be sure to play with this value and infer the internal
# behavior based on the output, for example the output sequence length.
BEAM_LENGTH?=30
# The current beam search is highly qualitative and introspective, not
# production. BEAM_DEBUG allows outputting extended beam-search introspection,
# such as outputting the greedy-decodings per each beam-item decode() operation,
# to observe the quality of term rankings considered by beam-search. Beam search
# is actually extremely engineer-able, and subject to statistical considerations
# as to what constitutes "best" words, so it is very useful to consider.
# This is an integer value to support levels of beam-debug: 0 means off.
BEAM_DEBUG?=0


.PHONY: clean
clean:
	rm -rf .venv
# Also blow away the temporary test model
	rm -rf ./trained_models/huck_test


# Rebuilds the venv. Currently the only known working combination is torch=2.3,
# which is required by torchtext=0.18, since torchtext is no longer supported
# and only supports torch 2.3.
#
# TODO: add requirements.txt to a pyproject.toml and use its build chain
# instead, even uv if possible.
#
# TODO: reduce dependencies and build time. This takes around 25mins on a good
# connection. Optimize with pip-caching in the devcontainer.
.PHONY: rebuild-venv
rebuild-venv:
	rm -rf .venv
	python3 -m venv .venv
	source .venv/bin/activate ; \
		pip3 install -r requirements.txt


# TODO: I modified the parameter-saving schedule to minimize disk usage, but
# this rate could be passed into the worker.
#
# train-novel implements a novel-reader dataset iterator. See the huckfinn
# example.
#
# Train supports two workflows:
# 1) set CONFIG_PATH to the path to a config for a model to train or load
# 2) pass nothing: by default this will train on a default dataset used as a
#    smoke test
.PHONY: train
train:
	source .venv/bin/activate && \
		LOG_LEVEL=$(LOG_LEVEL) \
		python training_driver.py -c $(CONFIG_PATH)


# check-outputs takes a fully trained model and its config, then generates
# output one token at a time by sampling. This is currently configured for
# beamsearch, whereby a beam of best-outputs are maintained as a source for
# appending new tokens.
#
# TODO: inference/prod has some additional parameters, i.e. beam-length when
# running beamsearch-inference. The scope of these could become so convoluted
# that breaking them out is weird, and they don't fully belong in the model
# config. Instead, the prod-time config could just be cli flags read with
# argparse.
#
# TODO: be sure you vary BEAM_LENGTH and don't just trust 'check-outputs' as a
# qualitative check. It may be a bug, or logical/behavioral bug, but a
# beam-length of 1000 caused the model to output nothing, such that the most
# likely output is shortened; a beam-length of 25 was closer to natural
# language. This result is consistent with the natural language property that
# most actual sequences are relatively low probability, whereas high probability
# sequences are not natural language, such as word loops. 
#
# TODO: evaluate beam for bugs. Large BEAM_LENGTH results in less sensible
# output and ultimately empty output; shorter lengths output more reasonable
# expressions. Something may be wrong with the summing of sequence probability.
#
# Set BEAM_LENGTH=1 for greedy decoding. As far as design, beam-length functions
# like a hyper-parameter in several ways.
.PHONY: check-outputs
check-outputs:
	source .venv/bin/activate && \
	BEAM_LENGTH=$(BEAM_LENGTH) \
	BEAM_DEBUG=$(BEAM_DEBUG) \
	LOG_LEVEL=$(LOG_LEVEL) \
		python check_outputs_driver.py -c $(CONFIG_PATH)


# Runs the unit-tests. Independently and fast-running
# test go here.
.PHONY: unit-tests
unit-tests:
	source .venv/bin/activate && \
		python -m pytest .


# train-test-model trains a vanilla seq2seq transformer on a very simply input
# dataset of repeating alphabetic sequences. This must be run before test-model
# to build the test model.
.PHONY: train-test-model
train-test-model:
	rm -f $(TEST_MODEL_DIR)/*
	source .venv/bin/activate && \
		LOG_LEVEL=$(LOG_LEVEL) \
		python training_driver.py -c $(TEST_MODEL_CONFIG_PATH)



# Runs the model-tests. These are both qualitative and quantitative tests as
# they run the entire model and beam_decode so the user can manually verify the
# output sequences are correct; thereby this confirms that the autoregressive
# structure of output is not violated, that experimental changes to the model
# work correctly on a very small example problem, and overall these tests catch
# model-based regressions in an end-to-end test. It is also a general purpose
# end-to-end test in the software sense.
#
# This recipe is the most useful tight-loop for development and model
# development. I learnt from implementing this that I wouldn't ever do without
# this form of test to ensure 'three points of contact' when doing model
# development or research. And yes, I would do this against all complaints from
# PMs about required setup time, and would gather their tears to add to my
# collection of PM tears. 
.PHONY: check-test-model-outputs
check-test-model-outputs:
	ls $(TEST_MODEL_DIR) | grep "independent_test_seq_final\.pt" \
		|| { echo "Missing $(TEST_MODEL_DIR)/independent_test_seq_final.pt. The test model must be built by running 'make train-test-model' before test-model." ; exit 1 ; }
	source .venv/bin/activate && \
		BEAM_LENGTH=2 \
		LOG_LEVEL=$(LOG_LEVEL) \
			python check_outputs_driver.py -c $(TEST_MODEL_CONFIG_PATH)


.PHONY: fmt
fmt:
	source .venv/bin/activate && \
		black .


###############################################################################
# These recipes are all experimental and not guaranteed up to date.
###############################################################################

# Converts ABLE fb-data format to line-based training file.
.PHONY: fb-data
fb-data:
	FB_DATA_PATH=./data/usatoday_fb.py \
		python ./scripts/fb_data_to_lines.py > ./data/fb_lines.txt