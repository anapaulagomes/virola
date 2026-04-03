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

# Run analysis views
uv run vir view clinical --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2025
uv run vir view demographic --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2025
uv run vir view temporal --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2025 --min-weeks 12

# Run SNF clustering (results saved automatically to virola.db)
uv run vir model snf --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2025 --snf-k 10 --n-clusters 200
uv run vir model snf --dataset data/interim/cleaned_*.parquet --year-min 2022 --year-max 2025 --snf-k 10 --n-clusters 200 --notes "run for paper"

# Inspect saved runs
uv run vir results list

# Build cluster profiles for a run (aggregated stats, temporal + demographic signatures)
uv run vir results build-profiles 1 --dataset data/interim/cleaned_*.parquet

# Generate LLM descriptions for all clusters in a run (skips already-described)
uv run vir results llm-descriptions 1
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
- `vir model snf --dataset DATASET [--year-min YEAR] [--year-max YEAR] [--snf-k INT] [--n-clusters INT] [--notes TEXT]` - Run SNF clustering, saves run to virola.db

#### Results
- `vir results list` - List all saved runs with metadata (city, parameters, scores)
- `vir results build-profiles RUN_ID --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Build aggregated cluster profiles (stats, temporal signature, demographic profile)
- `vir results explain RUN_ID --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Compute per-cluster explanations: primary view driver and central code (requires build-profiles first)
- `vir results llm-descriptions RUN_ID` - Generate LLM descriptions for all clusters in a run (skips already-described)

Use `vir COMMAND --help` for detailed options on any command.

### Development Mode

You will need `uv` to run this project. Install the dependencies with `uv sync`. Then, run `plotly_get_chrome`.
Use `UV_PROJECT_ENVIRONMENT=/home/<your-user>/virola/.venv uv sync` to create the virtual env dir in an specific location.

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
marimo run notebooks/view_clinical.py -- --dataset data.parquet --year-min 2022 --year-max 2025

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

### Validation files

The anchor validation used AESOP syndrome indicators, downloaded from the project's [repo](https://github.com/cidacslab/AESOP-Data-Documentation/).

Source:

* [code_list_arbovirus_apr2024.csv](https://raw.githubusercontent.com/cidacslab/AESOP-Data-Documentation/refs/heads/main/DataPipeline/documentation/code_list_arbovirus_apr2024.csv)
* [code_list_uri_apr2024.csv](https://raw.githubusercontent.com/cidacslab/AESOP-Data-Documentation/refs/heads/main/DataPipeline/documentation/code_list_uri_apr2024.csv)

### Indicative codes

An expert in our team have checked and labeled the most representative codes in the dataset
and classified them as indicative (sends an epidemiological signal) or not. This is useful
to filter out codes aren't adding much information to the approach. To create this CSV to
be labeled, you can run:

```bash
marimo run notebooks/non_indicative_codes.py
```

It will open a webpage. Use the Download button in the interface to download the files.
