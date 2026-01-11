# virola

Find public health threats to be monitored from your data

## Usage

The `vir` command provides organized access to all analysis tools through a unified CLI.

### Quick Start

```bash
# One-time setup
uv run vir setup terminologies  # Create terminology files
uv run vir setup embeddings     # Generate embeddings database

# Data preparation pipeline
uv run vir prepare clean --city "Carandaí"  # Optional: filter by city
uv run vir prepare features data/interim/cleaned_aesop_raw_dados_2017_20250223_20municipios_salvador.parquet

# Run analysis views
uv run vir view clinical --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2024
uv run vir view demographic --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2024
uv run vir view temporal --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2024 --min-weeks 12

# Train indicative codes model
# Download `indicative_codes_yes_no` from labeled data folder and place to `data/indicative_codes_yes_no.csv`
uv run vir model indicative-codes --input data/indicative_codes_yes_no.csv --output data/processed/indicative_probabilities_setfit.csv

# Run SNF clustering
uv run vir model snf --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2024 --snf-k 10 --n-clusters 200 --output data/clustering/
```

### CLI Reference

#### Setup Commands
- `vir setup terminologies` - Create terminology files from source data
- `vir setup embeddings [--model MODEL]` - Generate medical code embeddings database

#### Data Preparation
- `vir prepare clean [--city CITY] [--input FILE] [--output DIR]` - Clean and standardize raw data
- `vir prepare features DATASET [--level {records|diagnosis}]` - Extract features from cleaned data

#### Analysis Views
- `vir view clinical --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Semantic similarity analysis using medical embeddings
- `vir view demographic --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Demographic feature analysis
- `vir view temporal --dataset DATASET [--year-min YEAR] [--year-max YEAR] [--min-weeks INT]` - Time-series pattern analysis

#### Modeling
- `vir model indicative-codes --input FILE --output FILE` - Train indicative codes classifier
- `vir model snf --dataset DATASET --output DIR [--year-min YEAR] [--year-max YEAR] [--snf-k INT] [--n-clusters INT]` - Similarity Network Fusion clustering

Use `vir COMMAND --help` for detailed options on any command.

### Development Mode

For interactive development and exploration, use marimo's edit mode:

```bash
# Edit notebooks interactively
marimo edit notebooks/view_clinical.py
marimo edit notebooks/view_demographic.py
marimo edit vir/modeling/data_cleaning.py
```

### Alternative: Run Notebooks Directly

You can also run notebooks directly with marimo (bypassing the `vir` CLI):

```bash
# Run with arguments
marimo run notebooks/view_clinical.py -- --dataset data.parquet --year-min 2022 --year-max 2024

marimo run vir/modeling/data_cleaning.py -- --city "São Gonçalo do Amarante"
```

The `vir` CLI provides a more organized interface with better help text and validation.

### LLM Support

It is possible to describe the clusters using LLMs. For this, install: `llm install llm-ollama`
You will also need to have Ollama installed: https://ollama.com/download/

### Technical Details

#### Clinical View

Storage: DuckDB
Execution time: Insert clinical dimensions to the db 8 min
Visual inspection: [Embedding Atlas](https://github.com/apple/embedding-atlas)

## Notes

### Terminology

ABP008 is duplicated and it was renamed to ABP008-1 and ABP008-2 for distinction
in `cid10_ciap2_ab.csv`.
