.PHONY: test smoke

PYTHON ?= python3

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m src.smoke
