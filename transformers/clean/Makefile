# Description: this makefile is intended to mock out similar workflows to the
# desired api of a transformer for train/test.

SHELL:=/bin/bash

DATA_PATH?=./data/fb_lines.txt

CONFIG_PATH?=./deployment/default_config.jsonc

# Use DEBUG to view gritty model outputs, tensor dimensions, etc.
LOG_LEVEL?=WARNING

BEAM_LENGTH?=30


.PHONY: clean
clean:
	rm -rf .venv
# Also blow away the temporary test model
	rm -rf ./models/huck_test


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


# .PHONY: test-english-to-german
# test-english-to-german:
# 	source .venv/bin/activate && \
# 		python training_driver.py -c $(CONFIG_PATH)


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
	LOG_LEVEL=$(LOG_LEVEL) \
		python check_outputs_driver.py -c $(CONFIG_PATH)


.PHONY: unit-tests
unit-tests:
	source .venv/bin/activate && \
		python -m pytest .


.PHONY: fmt
fmt:
	source .venv/bin/activate && \
		black .


###############################################################################
# These recipes are all experimental and not guaranteed up to date.
###############################################################################

.PHONY: fb-data
fb-data:
	FB_DATA_PATH=./data/fb_data.py \
		python ./scripts/fb_data_to_lines.py > ./data/fb_lines.txt