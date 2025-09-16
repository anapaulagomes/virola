#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = virola
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Run tests
.PHONY: test
test:
	uv run pytest

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) vir/dataset.py

## Process city data pipeline (cleaning, processing, feature engineering)
## Usage: make city-pipeline CITY=<city_name> [LEVEL=records|diagnosis]
.PHONY: city-pipeline
city-pipeline: requirements
ifndef CITY
	$(error CITY is not set. Usage: make city-pipeline CITY=<city_name> [LEVEL=records|diagnosis])
endif
ifndef LEVEL
	$(eval LEVEL = records)
endif
	$(eval CITY_CLEAN := $(shell echo '$(CITY)' | tr '[:upper:]' '[:lower:]' | tr ' ' '_'))
	@echo "Running data pipeline for city: $(CITY) (filename: $(CITY_CLEAN)) at level: $(LEVEL)"
	@echo "Step 1: Running data cleaning..."
	uv run notebooks/02_data_cleaning.py -- --city $(CITY)
	@echo "Step 2: Running feature engineering ($(LEVEL) level)..."
	uv run $(PYTHON_INTERPRETER) -m vir.modeling.feature_engineering data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_$(CITY_CLEAN).parquet --level=$(LEVEL)
	@echo "Pipeline completed for city: $(CITY)"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
