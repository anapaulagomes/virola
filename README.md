# virola

Find public health threats to be monitored from your data

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (the project pins its Python version
in `.python-version`), then from the project root:

```bash
uv sync
uv run plotly_get_chrome      # required by kaleido for plotly figure export
cp .env.example .env          # adjust DATA_DIR / LLM_MODEL if needed
```

To place the virtualenv outside the project: `UV_PROJECT_ENVIRONMENT=/path/to/.venv uv sync`.

### Raw data

The pipeline expects an AESOP extraction at `data/raw/base_aesop_raw_extracao_01042026.parquet/`
(multi-file parquet). Schema, period, and the IBGE municipality filter are documented in
`data/raw/readme-base_aesop_raw_extracao_01042026.md`. Override the location with `DATA_DIR` and/or
`RAW_DATA_DIR` in `.env`.

The 10 cities targeted by the pipeline are listed in `bin/cities.txt`; the same names are accepted by
`vir pipeline run`.

### Terminologies and embeddings

Reference files (`references/RepositorioTerminologia_202506/tb_cid.csv`, `ciap2-cid10.csv`,
`abp_ciap2.csv`) are committed. Build the joined terminology table and the SapBERT embeddings database
once:

```bash
uv run vir setup terminologies
uv run vir setup embeddings
```

`setup embeddings` downloads `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` to `HF_HOME`.

### Optional: LLM descriptions

`vir results llm-descriptions` is optional. To enable it, install [Ollama](https://ollama.com/download/),
pull the model named in `LLM_MODEL` (default `llama3.2:1b`), and add the `llm-ollama` plugin:

```bash
ollama pull llama3.2:1b
uv run llm install llm-ollama
```

### Optional: indicative-codes labels

The pipeline ships with a labeled CSV (`references/indicative_codes_yes_no__2022_2025__90perc.csv`)
and uses it by default. Regenerate the labeling template only if you need a different time window —
see [Indicative codes](#indicative-codes) below.

## Usage

The `vir` command provides organized access to all analysis tools through a unified CLI. Helper
scripts under `bin/` wrap it for the all-cities case.

### Run all cities

`bin/cities.txt` drives both helpers. From a clean checkout (after Setup):

```bash
bin/run_pipeline.sh all                       # clean → process → views → SNF for every city
bin/enrich_latest_run.sh all                  # build-profiles + explain + llm-descriptions per city
uv run vir results list
```

Extra `vir pipeline run` flags pass through, e.g. `bin/run_pipeline.sh all --skip-clean --n-clusters 150`.
`bin/enrich_latest_run.sh` also accepts `--run-id`, `--dataset`, `--year-min`, `--year-max` to target a
specific run instead of the latest.

### Run a single city

```bash
uv run vir pipeline run "Belo Horizonte" --year-min 2022 --year-max 2024 \
    --snf-k 10 --n-clusters 200 --notes "run for paper"

uv run vir results list
bin/enrich_latest_run.sh                      # or call build-profiles / explain / llm-descriptions manually
```

`vir pipeline run` chains `clean → process → views (clinical, demographic, temporal) → model snf` for the
city. Run `uv run vir pipeline run --help` for the full list of options and the accepted cities. Without
arguments, `bin/enrich_latest_run.sh` enriches the most recent run in `data/virola.db`.

#### Step-by-step alternative

To run a single step (or substitute a custom dataset), the pipeline can be reproduced as:

```bash
uv run vir prepare clean --city "Belo Horizonte"
uv run vir prepare process --dataset data/interim/cleaned_*belo_horizonte*.parquet

uv run vir view clinical    --dataset belo_horizonte --year-min 2022 --year-max 2024
uv run vir view demographic --dataset belo_horizonte --year-min 2022 --year-max 2024
uv run vir view temporal    --dataset belo_horizonte --year-min 2022 --year-max 2024 --min-weeks 12

uv run vir model snf --dataset belo_horizonte --year-min 2022 --year-max 2024 \
    --snf-k 10 --n-clusters 200 --notes "run for paper"

uv run vir results build-profiles 1 --dataset belo_horizonte
uv run vir results explain 1 --dataset belo_horizonte    # requires build-profiles first
uv run vir results llm-descriptions 1                    # requires Ollama (see Setup)
```

Results from `vir model snf` (and `vir pipeline run`) are persisted to `data/virola.db` (DuckDB). Use the
`vir results` commands to inspect, enrich, or remove them.

### SLURM

Two SLURM scripts mirror the helpers above. Edit the `--mail-user` line at the top of each before
submitting; set `DATA_DIR` (in `.env` or via `--export`) when the shared filesystem is not the project
root.

```bash
sbatch bin/slurm_pipeline.sh                                          # full pipeline for every city
sbatch --export=ALL,EXTRA_ARGS="--skip-clean" bin/slurm_pipeline.sh

sbatch bin/slurm_ablation.sh                                          # ablation array, one task per city
sbatch --export=ALL,EXTRA_ARGS="--permutations 5000" bin/slurm_ablation.sh
```

The ablation array index range (`#SBATCH --array=0-9` in `bin/slurm_ablation.sh`) must match the number
of lines in `bin/cities.txt`.

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

#### Environment variables

The project reads a `.env` file at the project root if present. Recognized variables:

- `DATA_DIR` - override the default `./data` location (raw, interim, processed, runs, `virola.db` all live underneath)
- `RAW_DATA_DIR` - override `data/raw/` only
- `LLM_MODEL` - Ollama model used by `vir results llm-descriptions` (default `llama3.2:1b`)
- `HF_HOME` - cache directory for the SapBERT download
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
marimo edit vir/cleaning/data_cleaning.py
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

marimo run vir/cleaning/data_cleaning.py -- --city "Bauru"
```

The `vir` CLI provides a more organized interface with better help text and validation.

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
