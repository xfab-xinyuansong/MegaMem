.PHONY: install test self-test build check

install:
	python -m pip install -e ".[dev]"

test:
	pytest

self-test:
	python -m megamem.methods.dual_node
	python -m megamem.methods.token_ledger
	python -m megamem.methods.configs.isolation

build:
	python -m build

check: test self-test build
