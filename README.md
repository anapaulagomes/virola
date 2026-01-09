# virola

Find public health threats to be monitored from your data

## Usage

### Requirements

#### Dependencies

Create a terminology file:

```bash
uv run python vir/terminologies/terminologies.py  # adapt the duplicated ABP008 manually (see notes section)
```

Create a database with medical embeddings:

```bash
uv run python vir/terminologies/embeddings_model.py
```

Download `indicative_codes_yes_no` from labeled data folder and place to `data/indicative_codes_yes_no.csv`.
Then spin the command below to run the model:

```bash
uv run python notebooks/model_indicative_codes.py
```

#### LLM support

It is possible to describe the clusters using LLMs. For this, install: `llm install llm-ollama`
You will also need to have Ollama installed: https://ollama.com/download/

### Data cleaning

```
uv run vir/modeling/data_cleaning.py -- --city "Carandaí"
```

### Feature engineering

```
uv run python vir/modeling/feature_engineering.py data/interim/cleaned_aesop_raw...
```

### Views

#### Clinical

Storage: DuckDB
Execution time: Insert clinical dimensions to the db 8 min
Visual inspection: [Embedding Atlas](https://github.com/apple/embedding-atlas)

## Notes

### Terminology

ABP008 is duplicated and it was renamed to ABP008-1 and ABP008-2 for distinction
in `cid10_ciap2_ab.csv`.
