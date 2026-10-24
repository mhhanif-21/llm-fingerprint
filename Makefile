PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python
export MPLBACKEND := Agg
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export XDG_CACHE_HOME := $(CURDIR)/.cache
DATA_DIR ?= name-that-llm
OUTPUT_DIR ?= artifacts/eda
CONFIG ?= configs/baseline_svc.toml
TARGET ?= local
MODE ?= quick
RUN_ID ?=
RUN_DIR ?= artifacts/runs/$(RUN_ID)
TEST_ARGS ?= tests --ignore=tests/e2e

.PHONY: help setup setup-kaggle test lint format check-lines check-repo docs-check eda test-e2e train predict benchmark compare kaggle-run kaggle-status kaggle-fetch diagnose
help:
	@$(PYTHON) -c 'print("setup setup-kaggle test lint format check-lines check-repo docs-check eda train predict benchmark compare test-e2e kaggle-run kaggle-status kaggle-fetch diagnose")'
setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements-dev.lock
	$(PY) -m pip install --no-deps --no-build-isolation -e .
setup-kaggle:
	$(PY) -m pip install -e '.[kaggle]'
test:
	$(PY) -m pytest $(TEST_ARGS)
lint:
	$(PY) -m ruff check llmname llm_eda tests scripts
	$(PY) -m ruff format --check llmname llm_eda tests scripts
	$(PY) scripts/ci/check_cleanliness.py
format:
	$(PY) -m ruff check --fix llmname llm_eda tests scripts
	$(PY) -m ruff format llmname llm_eda tests scripts
check-lines:
	$(PY) scripts/ci/check_lines.py
check-repo:
	$(PY) scripts/ci/check_repo.py
docs-check:
	$(PY) scripts/ci/check_docs.py
eda:
	$(PY) -m llmname.eda --data-dir '$(DATA_DIR)' --output-dir '$(OUTPUT_DIR)' --skip-modeling $(if $(filter full,$(MODE)),,--skip-near-duplicates)
	$(PY) -m llmname.eda.extra --data-dir '$(DATA_DIR)' --output-dir '$(OUTPUT_DIR)/extra'
train:
	$(PY) -m llmname train --config '$(CONFIG)' --data-dir '$(DATA_DIR)' $(if $(filter smoke,$(MODE)),--smoke,)
predict:
	$(PY) -m llmname predict --run-dir '$(RUN_DIR)' --data-dir '$(DATA_DIR)'
benchmark:
	$(PY) -m llmname train --config '$(CONFIG)' --data-dir '$(DATA_DIR)' --smoke
compare:
	$(PY) -m llmname compare
test-e2e:
	LLMNAME_KAGGLE_E2E=$(if $(filter kaggle,$(TARGET)),1,0) $(PY) -m pytest tests/e2e/test_$(TARGET)_workflow.py -v -s
kaggle-run:
	$(PY) -m llmname kaggle-run --config '$(CONFIG)' $(if $(filter smoke,$(MODE)),--smoke,)
kaggle-status:
	$(PY) -m llmname kaggle-status --run-dir '$(RUN_DIR)'
kaggle-fetch:
	$(PY) -m llmname kaggle-fetch --run-dir '$(RUN_DIR)'
diagnose:
	$(PY) -m llmname diagnose --run-dir '$(RUN_DIR)'
