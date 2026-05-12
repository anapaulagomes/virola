# virola

Find public health threats to be monitored from your data.

`virola` is dataset-agnostic: a framework plus pluggable **adapters**. Two adapters ship today:

- **`datasus`** — Brazilian AESOP/DATASUS data (CID-10 + CIAP-2 + AB terminologies, 10 IBGE cities).
- **`aktin`** — German AKTIN emergency-department data (ICD-10-GM, country-level stratum).

Each adapter declares its own raw-format reader, terminology source, default embedding model,
skip patterns, and stratum metadata. Everything downstream of the canonical row
`(year, week, sex, age_group, code_type, code, quantity, terminology_id, terminology_label, [city])`
is shared.

## Setup (shared)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (the project pins its Python
version in `.python-version`), then from the project root:

```bash
uv sync
uv run plotly_get_chrome      # required by kaleido for plotly figure export
cp .env.example .env          # set DATA_DIR / RAW_DATA_DIR / VIROLA_ADAPTER as needed
```

To place the virtualenv outside the project: `UV_PROJECT_ENVIRONMENT=/path/to/.venv uv sync`.

### Environment variables

The project reads `.env` at the project root if present. Recognised variables:

- `DATA_DIR` — overrides the default `./data` location (raw, interim, processed, runs, `virola.db`).
- `RAW_DATA_DIR` — overrides `data/raw/` only. Point at your adapter's raw export.
- `VIROLA_ADAPTER` — default adapter for commands that take `--adapter`. CLI flag wins.
- `LLM_MODEL` — Ollama model used by `vir results llm-descriptions` (default `llama3.2:1b`).
- `HF_HOME` — cache directory for the SapBERT/XLMR downloads.
- `HF_FROM_LIMITED_ENV` — set to a non-empty value to force a manual HF snapshot download
  (SLURM / air-gapped). Empty value = regular environment.

### CLI shape

```text
# Adapter-level commands — require --adapter (or VIROLA_ADAPTER env var)
vir setup terminologies   --adapter <name>
vir setup embeddings      --adapter <name> [--model MODEL]

# Stratum-level commands — adapter is derived from the stratum registry
vir prepare clean         --stratum <name> [--input PATH] [--output DIR]
vir pipeline run          <stratum>        [--year-min ...] [--year-max ...] ...
vir validate ablation     --stratum <name>
vir results delete        --stratum <name>

# Canonical-row commands — adapter via env or explicit flag
vir prepare process       --dataset PATH   --adapter <name>
vir prepare indicative-template --dataset PATH --adapter <name>
vir view clinical         --dataset NAME   --adapter <name>
vir view demographic      --dataset NAME
vir view temporal         --dataset NAME
vir model snf             --dataset NAME   --adapter <name>
vir results build-profiles RUN_ID --dataset NAME
vir results explain        RUN_ID --dataset NAME --adapter <name>
vir results list
vir results llm-descriptions RUN_ID
```

`--city` is accepted as a deprecated alias for `--stratum` (emits a warning).

Use `uv run vir COMMAND --help` for detailed flag descriptions.

---

## DATASUS (Brazil)

### Raw data

The pipeline expects an AESOP extraction at `data/raw/base_aesop_raw_extracao_01042026.parquet/`
(multi-file parquet). Schema, period, and the IBGE municipality filter are documented in
`data/raw/readme-base_aesop_raw_extracao_01042026.md`. Override the location with `RAW_DATA_DIR`
in `.env`.

The 10 cities targeted by the pipeline are listed in `vir/adapters/datasus/bin/cities.txt`;
the same names are accepted by `vir pipeline run <stratum>`.

### One-time setup

Reference files (`vir/adapters/datasus/references/RepositorioTerminologia_202506/tb_cid.csv`,
`ciap2-cid10.csv`, `abp_ciap2.csv`) are committed. Build the joined terminology table and the
SapBERT embeddings database once:

```bash
uv run vir setup terminologies --adapter datasus
uv run vir setup embeddings    --adapter datasus
```

`setup embeddings` downloads `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` to `HF_HOME`.

### Run all cities

`vir/adapters/datasus/bin/cities.txt` drives both helpers:

```bash
vir/adapters/datasus/bin/run_pipeline.sh all                 # clean → process → views → SNF per city
vir/adapters/datasus/bin/enrich_latest_run.sh all            # build-profiles + explain + llm-descriptions
uv run vir results list
```

Extra `vir pipeline run` flags pass through, e.g.
`vir/adapters/datasus/bin/run_pipeline.sh all --skip-clean --n-clusters 150`.

### Run a single city

```bash
uv run vir pipeline run "Belo Horizonte" --year-min 2022 --year-max 2024 \
    --snf-k 10 --n-clusters 200 --notes "run for paper"

uv run vir results list
vir/adapters/datasus/bin/enrich_latest_run.sh
```

`vir pipeline run` chains `clean → process → views (clinical, demographic, temporal) → model snf`
and resolves the adapter (`datasus`) from the stratum name via the registry.

### Step-by-step alternative

```bash
uv run vir prepare clean   --stratum "Belo Horizonte"
uv run vir prepare process --dataset data/interim/cleaned_*belo_horizonte*.parquet \
                           --adapter datasus

uv run vir view clinical    --dataset belo_horizonte --adapter datasus \
                            --year-min 2022 --year-max 2024
uv run vir view demographic --dataset belo_horizonte --year-min 2022 --year-max 2024
uv run vir view temporal    --dataset belo_horizonte --year-min 2022 --year-max 2024 --min-weeks 12

uv run vir model snf --dataset belo_horizonte --adapter datasus \
    --year-min 2022 --year-max 2024 --snf-k 10 --n-clusters 200 --notes "run for paper"

uv run vir results build-profiles 1 --dataset belo_horizonte
uv run vir results explain 1 --dataset belo_horizonte --adapter datasus
uv run vir results llm-descriptions 1                   # requires Ollama (see Optional below)
```

### SLURM

Two SLURM scripts mirror the helpers above; edit the `--mail-user` line at the top of each before
submitting and set `DATA_DIR` (in `.env` or via `--export`) when the shared filesystem is not the
project root.

```bash
sbatch vir/adapters/datasus/bin/slurm_pipeline.sh                                  # full pipeline, every city
sbatch --export=ALL,EXTRA_ARGS="--skip-clean" vir/adapters/datasus/bin/slurm_pipeline.sh

sbatch vir/adapters/datasus/bin/slurm_ablation.sh                                  # ablation array, one task per city
sbatch --export=ALL,EXTRA_ARGS="--permutations 5000" vir/adapters/datasus/bin/slurm_ablation.sh
```

The ablation array index range (`#SBATCH --array=0-9` in `slurm_ablation.sh`) must match the
number of lines in `vir/adapters/datasus/bin/cities.txt`.

---

## AKTIN (Germany)

The AKTIN adapter treats Germany as a single stratum (`"germany"`). Per-encounter raw data lives
in `*_result/case_data.txt` + `diag_data.txt` pairs (tab-separated). The adapter pools every site
into one cleaned interim parquet.

### Raw data

Each AKTIN export ships per-hospital folders/zips. Either:

- a directory of unzipped `<N>_result/` subdirectories under `RAW_DATA_DIR`, **or**
- a directory containing a single site's `case_data.txt` and `diag_data.txt`.

Set `RAW_DATA_DIR` in `.env` (absolute path, or leave blank to use `data/raw/`).

### One-time setup

```bash
uv run vir setup terminologies --adapter aktin       # parses BfArM ICD-10-GM 2025 (vendored)
uv run vir setup embeddings    --adapter aktin       # downloads SapBERT-UMLS-XLMR (~2.2 GB)
```

The BfArM ICD-10-GM 2025 classification file is vendored at
`vir/adapters/aktin/references/icd10gm2025syst_kodes.txt`. AKTIN's default embedding model is
`cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR` (multilingual; embeds German labels
reasonably).

### Run the pipeline

```bash
uv run vir pipeline run germany --year-min 2018 --year-max 2023 \
    --snf-k 10 --n-clusters 200 --notes "first aktin run"

uv run vir results list
uv run vir results build-profiles 1 --dataset germany
uv run vir results explain 1 --dataset germany --adapter aktin
```

`vir pipeline run` derives the adapter (`aktin`) from the stratum `germany` via the registry and
threads it through every step.

### Step-by-step alternative

```bash
uv run vir prepare clean   --stratum germany                                    # adapter derived
uv run vir prepare process --dataset data/interim/cleaned_aktin_germany.parquet \
                           --adapter aktin

uv run vir view clinical    --dataset germany --adapter aktin \
                            --year-min 2018 --year-max 2023
uv run vir view demographic --dataset germany --year-min 2018 --year-max 2023
uv run vir view temporal    --dataset germany --year-min 2018 --year-max 2023 --min-weeks 12

uv run vir model snf --dataset germany --adapter aktin \
    --year-min 2018 --year-max 2023 --snf-k 10 --n-clusters 200

uv run vir results build-profiles 1 --dataset germany
uv run vir results explain 1 --dataset germany --adapter aktin
```

### Indicative-codes template

AKTIN ships **without** an epidemiologist-labeled CSV. To generate the top-volume template for
labeling:

```bash
uv run vir prepare indicative-template \
    --dataset data/interim/cleaned_aktin_germany.parquet \
    --adapter aktin \
    --coverage 0.9
# writes data/processed/indicative_codes_template_aktin.csv
```

Until labels are supplied, AKTIN runs effectively with `--no-indicative` (only Z-chapter codes
are filtered via `adapter.skip_patterns`).

### Exploratory data analysis

A starter notebook lives at `vir/adapters/aktin/notebooks/eda_aktin.py` (shape, demographics,
ICD-10-GM coverage, monthly/weekly temporal distribution):

```bash
uv run marimo edit vir/adapters/aktin/notebooks/eda_aktin.py
```

---

## Adding a new adapter

The contract is the abstract base class `vir/adapters/base.py::Adapter`. Concrete adapters live
under `vir/adapters/<name>/` and own their cleaning, terminology, references, notebooks, bin
scripts, and tests. Each adapter declares:

- `name`, `default_model`, `terminology_csv_filename` — class attributes.
- `clean(raw_path, output_path, stratum, **opts)` — read raw → write canonical interim parquet.
- `build_terminology_labels()` — return `(code, code_type, description)` DataFrame.
- `stratum_metadata(stratum)` — `{city, region, population_range}`.
- `cleaned_filename(stratum)` — interim filename convention.
- `strata`, `skip_patterns`, `code_types` — properties.
- `indicative_codes_file` — optional `Path`, defaults to `None`.

Register the instance in `vir/adapters/__init__.py::ADAPTERS`. The registry detects stratum
collisions across adapters at import time. See `vir/adapters/datasus/` and `vir/adapters/aktin/`
for the two reference implementations.

---

## Optional

### LLM descriptions

`vir results llm-descriptions` is optional. To enable it, install
[Ollama](https://ollama.com/download/), pull the model named in `LLM_MODEL` (default
`llama3.2:1b`), and add the `llm-ollama` plugin:

```bash
ollama pull llama3.2:1b
uv run llm install llm-ollama
```

### Development

```bash
make test            # or: uv run pytest
make lint            # ruff format --check && ruff check
```

Interactive marimo edit mode for exploration:

```bash
uv run marimo edit notebooks/view_clinical.py                              # framework
uv run marimo edit vir/adapters/datasus/notebooks/view_demographic.py      # DATASUS
uv run marimo edit vir/adapters/aktin/notebooks/eda_aktin.py               # AKTIN
```

Two DATASUS notebook subfolders are organised by purpose:

- `vir/adapters/datasus/notebooks/validation/` — paper-grade validation (ablation, cross-city,
  reference syndromes, temporal C). The ablation notebook is also exposed as `vir validate
  ablation`.
- `vir/adapters/datasus/notebooks/sanity_checks/` — internal exploratory checks per view
  (`sc_clinical`, `sc_demographic`, `sc_temporal`, feature transformation comparison,
  temporal distance comparison). Not part of the paper's evidence base.

## Notes

### Terminology (DATASUS)

ABP008 is duplicated in the official AB terminology source
(<https://integracao.esusab.ufsc.br/ledi/documentacao/estrutura_arquivos/dicionario-fai.html#listaciapcondicaoavaliada>),
mapping to two distinct conditions: Beribéri (ABP008-1) and Desnutrição (ABP008-2).
The suffixes `-1` / `-2` are added in the reference files (`abp_ciap2.csv`,
`terminology_mapping.csv`) for disambiguation only — the raw data always carries the base code
`ABP008` without any suffix.

### AESOP validation files (DATASUS)

The anchor validation used AESOP syndrome indicators, downloaded from the project's
[repo](https://github.com/cidacslab/AESOP-Data-Documentation/):

- [code_list_arbovirus_apr2024.csv](https://raw.githubusercontent.com/cidacslab/AESOP-Data-Documentation/refs/heads/main/DataPipeline/documentation/code_list_arbovirus_apr2024.csv)
- [code_list_uri_apr2024.csv](https://raw.githubusercontent.com/cidacslab/AESOP-Data-Documentation/refs/heads/main/DataPipeline/documentation/code_list_uri_apr2024.csv)

### Indicative codes (DATASUS)

An expert reviewed the codes covering the top 90% of record volume and labeled each as indicative
(carries an epidemiological signal) or non-indicative. Codes labeled non-indicative are excluded
from the analysis; codes not yet reviewed (outside the top 90%, or reviewed but unlabeled) are
retained.

To generate (or regenerate) the labeling template for a different time window:

```bash
uv run vir prepare indicative-template \
    --dataset data/interim/cleaned_*_all_cities.parquet \
    --adapter datasus \
    --coverage 0.9
```
