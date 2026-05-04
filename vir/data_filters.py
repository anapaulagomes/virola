import polars as pl

from vir.config import MIN_WEEKS


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


def filter_min_temporal_coverage(df, min_weeks=None, *, by_year: bool = True):
    """Filter terminology codes by minimal temporal coverage.

    Counts distinct weeks. The prepared parquet is aggregated to
    one row per (year, week, sex, age_group, code_type, code), so a single
    (terminology_id, year, week) typically spans multiple demographic rows.

    Args:
        df: DataFrame with columns ``terminology_id``, ``year``, ``week``.
        min_weeks: Minimum distinct weeks required. ``None`` falls back to
            ``MIN_WEEKS``; ``0`` is honoured as "no cutoff".
        by_year: When ``True`` (default), the threshold is applied per
            ``(terminology_id, year)``. When ``False``, it is applied once
            across the full period per ``terminology_id``.
    """
    if min_weeks is None:
        min_weeks = MIN_WEEKS

    if not by_year:
        more_than_min_weeks = (
            df.group_by("terminology_id")
            .agg(pl.col("week").n_unique().alias("week_count"))
            .filter(pl.col("week_count").ge(min_weeks))
        )
        return df.filter(
            pl.col("terminology_id").is_in(more_than_min_weeks["terminology_id"].implode().first())
        )

    passing = (
        df.group_by(["year", "terminology_id"])
        .agg(pl.col("week").n_unique().alias("week_count"))
        .filter(pl.col("week_count").ge(min_weeks))
        .select("year", "terminology_id")
    )
    return df.join(passing, on=["year", "terminology_id"], how="semi")


def filter_temporal_quantiles(df: pl.DataFrame, quantile: float = 0.20) -> pl.DataFrame:
    """Keep terminology codes whose temporal coverage and weekly mean quantity
    both meet a city-specific quantile threshold (default: P20).

    For each ``terminology_id`` in ``df`` (assumed to be already scoped to a
    single city — the prepared parquet files are per-city), this computes:

    - ``nonzero_weeks``: number of distinct (year, week) pairs the code appears in
    - ``weekly_mean``: total quantity divided by ``nonzero_weeks``

    A code is kept only if **both** ``nonzero_weeks`` and ``weekly_mean`` are
    greater than or equal to the corresponding ``quantile`` of the per-code
    distribution within the city. This is intended as an alternative to the
    fixed ``MIN_WEEKS`` cutoff used by ``filter_min_temporal_coverage``.

    Args:
        df: DataFrame with columns ``terminology_id``, ``year``, ``week`` and
            ``quantity``.
        quantile: Quantile threshold applied independently to both metrics.
            Defaults to ``0.20`` (P20).

    Returns:
        Filtered DataFrame containing only rows whose ``terminology_id`` passes
        both thresholds.
    """
    code_metrics = (
        df.group_by("terminology_id")
        .agg(
            pl.struct("year", "week").n_unique().alias("nonzero_weeks"),
            pl.col("quantity").sum().alias("total_quantity"),
        )
        .with_columns(
            (pl.col("total_quantity") / pl.col("nonzero_weeks")).alias("weekly_mean"),
        )
    )
    nonzero_weeks_threshold = code_metrics["nonzero_weeks"].quantile(quantile)
    weekly_mean_threshold = code_metrics["weekly_mean"].quantile(quantile)
    codes_to_keep = code_metrics.filter(
        (pl.col("nonzero_weeks") >= nonzero_weeks_threshold)
        & (pl.col("weekly_mean") >= weekly_mean_threshold)
    )["terminology_id"]
    return df.filter(pl.col("terminology_id").is_in(codes_to_keep.implode().first()))


def filter_skip_patterns(df: pl.DataFrame, patterns: list[tuple]) -> pl.DataFrame:
    if not patterns:
        return df
    mask = pl.lit(False)
    for code_type, regex in patterns:
        code_matches = pl.col("code").str.contains(regex)
        if code_type is None:
            mask = mask | code_matches
        else:
            mask = mask | ((pl.col("code_type") == code_type) & code_matches)
    return df.filter(~mask)
