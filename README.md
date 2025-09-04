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
