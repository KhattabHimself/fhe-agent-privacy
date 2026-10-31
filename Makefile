# ============================================================
#  FHE-Agent-Privacy — Build & Experiment Automation
# ============================================================
#
#  "make all"    — Full pipeline: data → benchmarks → leakage →
#                  figures → demo.  Expect 30–60 minutes.
#
#  "make quick"  — Toy data + fast benchmarks + figures for
#                  rapid iteration (~5 minutes).
#
#  Run "make venv" once before anything else.
#
# ============================================================

SHELL := /bin/bash

# Make the repo root importable (scripts use "from src... import ...")
export PYTHONPATH := $(CURDIR)

# Python interpreter — use the venv binary directly so every
# recipe works without manual activation.
VENV_DIR := venv
ifeq ($(OS),Windows_NT)
    PYTHON := $(VENV_DIR)/Scripts/python.exe
    PIP    := $(VENV_DIR)/Scripts/pip.exe
else
    PYTHON := $(VENV_DIR)/bin/python
    PIP    := $(VENV_DIR)/bin/pip
endif

RESULTS_DIR  := results
FIGURES_DIR  := $(RESULTS_DIR)/figures

# ============================================================
#  Environment
# ============================================================

venv:
	python3 -m venv $(VENV_DIR) || python -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	@echo "Virtual environment created at $(VENV_DIR)/."

check:
	$(PYTHON) setup_check.py

# ============================================================
#  Data
# ============================================================

data:
	$(PYTHON) data/generate_all.py

# ============================================================
#  Experiments
# ============================================================

benchmarks-pir:
	$(PYTHON) experiments/run_benchmarks.py --pir-only

benchmarks-fhe:
	$(PYTHON) experiments/run_benchmarks.py --fhe-only

benchmarks-pipeline:
	$(PYTHON) experiments/run_benchmarks.py

benchmarks-memory:
	$(PYTHON) experiments/run_memory_benchmarks.py

benchmarks-systems:
	$(PYTHON) experiments/run_systems_analysis.py

benchmarks: benchmarks-pir benchmarks-fhe benchmarks-pipeline benchmarks-memory benchmarks-systems

leakage:
	$(PYTHON) experiments/run_leakage_analysis.py

# ============================================================
#  Figures
# ============================================================

figures:
	$(PYTHON) experiments/visualize_feasibility.py
	$(PYTHON) experiments/visualize_threat_model.py
	$(PYTHON) experiments/visualize_precision.py

# ============================================================
#  Demo
# ============================================================

demo:
	$(PYTHON) experiments/run_prototype_demo.py

# ============================================================
#  Tests
# ============================================================

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# ============================================================
#  Pipelines
# ============================================================

all: data benchmarks leakage figures demo
	@echo ""
	@echo "=== ALL DONE ==="
	@echo "Results: $(RESULTS_DIR)/"

quick:
	$(PYTHON) data/generate_all.py
	$(PYTHON) experiments/run_benchmarks.py --small
	$(PYTHON) experiments/visualize_feasibility.py

# ============================================================
#  Clean
# ============================================================

clean:
	rm -rf $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.txt
	rm -rf $(FIGURES_DIR)/*.pdf $(FIGURES_DIR)/*.png
	find . -path ./$(VENV_DIR) -prune -o -type d -name __pycache__ -print | xargs rm -rf
	@echo "Cleaned generated files. venv preserved."

clean-all: clean
	rm -rf $(VENV_DIR)
	@echo "Cleaned everything including venv."

# ============================================================
#  Help
# ============================================================

help:
	@echo "FHE-Agent-Privacy Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make venv                 Create virtualenv + install deps (run once)"
	@echo "  make check                Verify all dependencies are importable"
	@echo ""
	@echo "Pipelines:"
	@echo "  make all                  data → benchmarks → leakage → figures → demo (30-60 min)"
	@echo "  make quick                Toy data + fast benchmarks (~5 min)"
	@echo ""
	@echo "Individual stages:"
	@echo "  make data                 Generate synthetic datasets"
	@echo "  make benchmarks           All benchmarks (PIR + FHE + pipeline + memory + systems)"
	@echo "  make benchmarks-pir       PIR benchmarks only"
	@echo "  make benchmarks-fhe       FHE benchmarks only"
	@echo "  make benchmarks-pipeline  Full pipeline benchmarks"
	@echo "  make benchmarks-memory    Memory profiling"
	@echo "  make benchmarks-systems   Systems friction analysis"
	@echo "  make leakage              Leakage metrics + intent-reconstruction attacks"
	@echo "  make figures              Generate all plots"
	@echo "  make demo                 End-to-end prototype demo"
	@echo "  make test                 Run the pytest suite"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                Remove generated results (keeps venv)"
	@echo "  make clean-all            Remove everything including venv"

.PHONY: venv check data \
        benchmarks benchmarks-pir benchmarks-fhe benchmarks-pipeline \
        benchmarks-memory benchmarks-systems leakage \
        figures demo test all quick clean clean-all help
