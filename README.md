# virola

Find public health threats to be monitored from your data

## Usage

The `vir` command provides organized access to all analysis tools through a unified CLI.

### Quick Start

```bash
uv run vir setup terminologies
uv run vir setup embeddings

uv run vir pipeline run "Belo Horizonte" --year-min 2022 --year-max 2024 --snf-k 10 --n-clusters 200 --notes "run for paper"

uv run vir results list
uv run vir results build-profiles 1 --dataset belo_horizonte
uv run vir results explain 1 --dataset belo_horizonte
uv run vir results llm-descriptions 1
```

`vir pipeline run` chains `clean → process → views (clinical, demographic, temporal) → model snf` for a single
city. Run `uv run vir pipeline run --help` for the full list of options and the available cities.

#### Manual alternative

If you want to run a single step (or substitute a custom dataset), the pipeline can be reproduced as:

```bash
uv run vir prepare clean --city "Belo Horizonte"
uv run vir prepare process --dataset data/interim/cleaned_*belo_horizonte*.parquet

uv run vir view clinical    --dataset belo_horizonte --year-min 2022 --year-max 2024
uv run vir view demographic --dataset belo_horizonte --year-min 2022 --year-max 2024
uv run vir view temporal    --dataset belo_horizonte --year-min 2022 --year-max 2024 --min-weeks 12

uv run vir model snf --dataset belo_horizonte --year-min 2022 --year-max 2024 --snf-k 10 --n-clusters 200 --notes "run for paper"
```

Results from `vir model snf` (and `vir pipeline run`) are persisted to `data/virola.db` (DuckDB). Use the
`vir results` commands to inspect, enrich, or remove them.

### CLI Reference

#### Setup Commands

- `vir setup terminologies` - Create terminology files from source data
- `vir setup embeddings [--model MODEL]` - Generate medical code embeddings database

#### Data Preparation

- `vir prepare clean [--city CITY] [--input FILE] [--output DIR] [--force]` - Clean and standardize raw data
- `vir prepare process --dataset DATASET` - Aggregate, join terminology, and filter codes on a cleaned interim dataset

#### Analysis Views

- `vir view clinical --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Semantic similarity analysis using medical embeddings
- `vir view demographic --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Demographic feature analysis
- `vir view temporal --dataset DATASET [--year-min YEAR] [--year-max YEAR] [--min-weeks INT]` - Time-series pattern analysis

#### Modeling

- `vir model snf --dataset DATASET [--year-min YEAR] [--year-max YEAR] [--snf-k INT] [--n-clusters INT] [--notes TEXT] [--no-indicative]` - Run SNF clustering, saves run to `data/virola.db`
- `vir model indicative-codes --input FILE --output FILE` - Train the indicative-codes classifier (SetFit)

#### Pipeline

- `vir pipeline run CITY [--year-min YEAR] [--year-max YEAR] [--min-weeks INT] [--snf-k INT] [--n-clusters INT] [--skip-clean] [--no-indicative] [--notes TEXT]` - Run the full pipeline (`clean → process → views → model snf`) for a single city

#### Results

- `vir results list [--limit N]` - List all saved runs with metadata (city, parameters, scores)
- `vir results build-profiles RUN_ID --dataset DATASET [--year-min YEAR] [--year-max YEAR]` - Build aggregated cluster profiles (stats, temporal signature, demographic profile)
- `vir results explain RUN_ID --dataset DATASET [--year-min YEAR] [--year-max YEAR] [--min-weeks INT]` - Compute per-cluster explanations: primary view driver and central code (requires build-profiles first)
- `vir results llm-descriptions RUN_ID` - Generate LLM descriptions for all clusters in a run (skips already-described)
- `vir results delete (--id RUN_ID | --city CITY) [--yes]` - Delete a saved run (or every run for a city) including assignments, profiles, sweep rows, and the parquet sidecar

#### Validation

- `vir validate ablation [--city CITY] [--permutations INT] [--output DIR] [--parallel/--no-parallel]` - Run the ablation study used in the paper

Use `vir COMMAND --help` for detailed options on any command.

### Development Mode

You will need `uv` to run this project. Install the dependencies with `uv sync`, then run `uv run plotly_get_chrome`
(needed by `kaleido` to export figures with plotly).
Use `UV_PROJECT_ENVIRONMENT=/home/<your-user>/virola/.venv uv sync` to create the virtual env dir in a specific location.

#### Environment variables

The project reads a `.env` file at the project root if present. Recognized variables:

- `DATA_DIR` - override the default `./data` location (raw, interim, processed, runs, `virola.db` all live underneath)
- `VIROLA_SKIP_PATTERNS` - semicolon-separated `code_type|regex` entries used to drop codes during preparation
  (default drops ICD-10 `Z*` codes — see `vir/config.py`)

#### Tests

```bash
make test            # or: uv run pytest
make lint            # ruff format --check && ruff check
```

#### Interactive notebooks

For exploration use marimo's edit mode:

```bash
marimo edit notebooks/view_clinical.py
marimo edit notebooks/view_demographic.py
marimo edit vir/modeling/data_cleaning.py
```

Two notebook folders are organized by purpose:

- `notebooks/validation/` — paper-grade validation (ablation, cross-city, reference syndromes, temporal C). The ablation
  notebook is also exposed as `vir validate ablation`.
- `notebooks/sanity_checks/` — internal exploratory checks per view (`sc_clinical`, `sc_demographic`, `sc_temporal`,
  feature transformation comparison, temporal distance comparison). Not part of the paper's evidence base.

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

ABP008 is duplicated in the official AB terminology source
(https://integracao.esusab.ufsc.br/ledi/documentacao/estrutura_arquivos/dicionario-fai.html#listaciapcondicaoavaliada),
mapping to two distinct conditions: Beribéri (ABP008-1) and Desnutrição (ABP008-2).
The suffixes `-1` / `-2` are added in the reference files (`abp_ciap2.csv`,
`terminology_mapping.csv`) for disambiguation only — the raw data always carries
the base code `ABP008` without any suffix.

### Validation files

The anchor validation used AESOP syndrome indicators, downloaded from the project's [repo](https://github.com/cidacslab/AESOP-Data-Documentation/).

Source:

* [code_list_arbovirus_apr2024.csv](https://raw.githubusercontent.com/cidacslab/AESOP-Data-Documentation/refs/heads/main/DataPipeline/documentation/code_list_arbovirus_apr2024.csv)
* [code_list_uri_apr2024.csv](https://raw.githubusercontent.com/cidacslab/AESOP-Data-Documentation/refs/heads/main/DataPipeline/documentation/code_list_uri_apr2024.csv)

### Indicative codes

An expert in our team reviewed the codes covering the top 90% of record volume and
labeled each as indicative (carries an epidemiological signal) or non-indicative.
Codes explicitly labeled non-indicative are excluded from the analysis; codes not yet
reviewed (outside the top 90%, or reviewed but unlabeled) are retained. This filters
out administrative and procedural codes while preserving coverage of rare conditions.
To create this CSV to be labeled, you can run:

```bash
marimo run notebooks/non_indicative_codes.py
```

It will open a webpage. Use the Download button in the interface to download the files.
