
SHELL := /bin/bash


# Rebuilds the venv. Currently the only known working combination is torch=2.3,
# which is required by torchtext=0.18, since torchtext is no longer supported
# and only supports torch 2.3
.PHONY: rebuild-venv
rebuild-venv:
	deactivate || true
	rm -rf venv
	python3 -m venv venv
	source venv/bin/activate ; \
	pip3 install -r requirements_torch_pin.txt