# Description: this makefile is intended to mock out similar workflows to the
# desired api of a transformer for train/test.


SHELL:=/bin/bash
# TODO: create a very small test-dataset
DATA_PATH?=../attn/data/huckfinn.txt
CONFIG_PATH?=./depoloyment/default_config.jsonc
LOG_LEVEL?=WARNING


# Rebuilds the venv. Currently the only known working combination is torch=2.3,
# which is required by torchtext=0.18, since torchtext is no longer supported
# and only supports torch 2.3
#
# TODO: add requirements.txt to a pyproject.toml and use its build chain instead,
# even uv if possible.
#
# TODO: reduce dependencies and build time. This takes around 25mins on a good
# connection. Optimize with pip-caching in the devcontainer.
.PHONY: rebuild-venv
rebuild-venv:
	rm -rf .venv
	python3 -m venv .venv
	source .venv/bin/activate ; \
		pip3 install -r requirements.txt


# TODO: continue here, ensure that train and run are both compact and
# well-encapsulated and invertible, etc. Create a super simple training dataset
# of a few hundreds examples as a simple sw-test training set, such that this
# can be used to train, plot training, and then reload the model for inference.
#
# Train supports two workflows:
# 1) set CONFIG_PATH to the path to a config for a model to train or load
# 2) pass nothing: by default this will train on a default dataset used as a
#    smoke test
.PHONY: train
train:
	source venv/bin/activate && \
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
.PHONY: check-outputs
check-outputs:
	source .venv/bin/activate && \
	BEAM_LENGTH=50 \
		python check_outputs.py -c $(CONFIG_PATH)


.PHONY: clean
clean:
	rm -rf .venv
