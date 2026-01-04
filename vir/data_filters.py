import polars as pl


def filter_most_representative_rows(df, threshold=0.95, group_by=None, frequency_column=None):
    if not group_by:
        group_by = ["terminology_label"]
    if not frequency_column:
        frequency_column = "quantity"
    return (
        df.group_by(group_by)
        .agg(pl.col(frequency_column).sum())
        .sort(frequency_column, descending=True)
        .with_columns(
            [
                (pl.col(frequency_column).cum_sum() / pl.col(frequency_column).sum()).alias(
                    "cum_pct"
                ),
            ]
        )
        # keep rows where the previous cumulative sum was less than the threshold
        # this ensures we include the row that crosses the threshold
        .filter(pl.col("cum_pct").shift(1, fill_value=0) < threshold)
    )
