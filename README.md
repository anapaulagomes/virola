# virola

Find public health threats to be monitored from your data

## Usage

### Data cleaning

```
uv run notebooks/02_data_cleaning.py -- --city "Carandaí"
```

### Feature engineering

```
uv run python vir/modeling/feature_engineering.py data/interim/cleaned_aesop_raw...
```

### Clustering

HDBSCAN

```
python vir/modeling/train.py data/processed/processed_optimized_marechal_thaumaturgo.parquet --model hdbscan --label "04092025"
```

DBSCAN

```
python vir/modeling/train.py data/processed/processed_optimized_marechal_thaumaturgo.parquet --model dbscan --label "04092025"
```

### Terminology

ABP008 is duplicated and it was renamed to ABP008-1 and ABP008-2 for distinction
in `cid10_ciap2_ab.csv`.

### LLM support

It is possible to describe the clusters using LLMs. For this, install: llm install llm-ollama
You will also need to have Ollama installed: https://ollama.com/download/
