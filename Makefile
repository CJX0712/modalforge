.PHONY: venv install test demo bench ruff clean

VENV ?= .venv
PY := $(VENV)/Scripts/python.exe
ifeq ($(OS),Windows_NT)
	PY := $(VENV)/Scripts/python.exe
else
	PY := $(VENV)/bin/python
endif

venv:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

install: venv

test:
	$(PY) -m pytest -q -W ignore::UserWarning

demo:
	$(PY) -m modalforge.examples.run_demo

bench:
	$(PY) -m modalforge.cli --out benchmark.json

ruff:
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

clean:
	rm -rf $(VENV) .pytest_cache __pycache__ */__pycache__
