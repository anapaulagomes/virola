"""Persistence helpers for SNF run results.

Responsible for:
- Resolving city/region metadata from dataset_name via IBGE_CITY_MAP
- Saving a complete run (run metadata + cluster assignments) to virola.db
- Building cluster profiles on demand from raw data
"""

import json

import numpy as np
import polars as pl

from vir.config import DATA_DIR, logger
from vir.db import (
    insert_cluster_assignments,
    insert_run,
    insert_run_multi_k,
    upsert_cluster_profile,
)
from vir.features import IBGE_CITY_MAP

# ---------------------------------------------------------------------------
# City metadata resolution
# ---------------------------------------------------------------------------


def _city_metadata(dataset_name: str) -> dict:
    """Extract city, region and population_range from dataset_name.

    Matches by normalising the city name from IBGE_CITY_MAP against the
    dataset_name string (lowercase, underscored). Returns None values
    when no match is found so the run is still recorded.
    """
    for entry in IBGE_CITY_MAP.values():
        normalised = entry["city"].lower().replace(" ", "_")
        if normalised in dataset_name.lower():
            return {
                "city": entry["city"],
                "region": entry["region"],
                "population_range": entry["population_range"],
            }
    logger.warning(f"Could not resolve city metadata for dataset '{dataset_name}'")
    return {"city": None, "region": None, "population_range": None}


# ---------------------------------------------------------------------------
# Save a complete SNF run
# ---------------------------------------------------------------------------


def save_run(
    *,
    dataset_name: str,
    year_min: int,
    year_max: int,
    snf_k: int,
    n_clusters: int,
    fused_labels: np.ndarray,
    common_ids: list[int],
    indicative_enabled: bool = True,
    silhouette_score: float | None = None,
    affinity_zscore: float | None = None,
    notes: str | None = None,
    created_at: str | None = None,
    df: pl.DataFrame | None = None,
    n_codes_clinical: int | None = None,
    n_codes_temporal: int | None = None,
    n_codes_demographic: int | None = None,
    n_missing_temporal: int | None = None,
    n_missing_demographic: int | None = None,
    cluster_largest: int | None = None,
    cluster_median: float | None = None,
    n_singletons: int | None = None,
    pct_singletons: float | None = None,
    multi_k_results: list[dict] | None = None,
    suggested_k: int | None = None,
) -> int:
    """Persist a complete SNF run to the database.

    Inserts one row in `runs` and one row per code in `cluster_assignments`.
    Optionally persists view coverage stats, cluster distribution stats, and
    multi-K comparison results.

    Returns:
        The run_id of the inserted run.
    """
    meta = _city_metadata(dataset_name)
    n_codes_common = len(common_ids)

    run_id = insert_run(
        dataset_name=dataset_name,
        city=meta["city"],
        region=meta["region"],
        population_range=meta["population_range"],
        year_min=year_min,
        year_max=year_max,
        snf_k=snf_k,
        n_clusters=n_clusters,
        indicative_enabled=indicative_enabled,
        silhouette_score=silhouette_score,
        affinity_zscore=affinity_zscore,
        notes=notes,
        created_at=created_at,
        n_codes_clinical=n_codes_clinical,
        n_codes_temporal=n_codes_temporal,
        n_codes_demographic=n_codes_demographic,
        n_codes_common=n_codes_common,
        n_missing_temporal=n_missing_temporal,
        n_missing_demographic=n_missing_demographic,
        cluster_largest=cluster_largest,
        cluster_median=cluster_median,
        n_singletons=n_singletons,
        pct_singletons=pct_singletons,
        suggested_k=suggested_k,
    )

    assignments = [
        {"terminology_id": int(tid), "cluster_id": int(label)}
        for tid, label in zip(common_ids, fused_labels)
    ]
    insert_cluster_assignments(run_id, assignments)

    if multi_k_results:
        insert_run_multi_k(run_id, multi_k_results)

    if df is not None:
        runs_dir = DATA_DIR / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = runs_dir / f"{run_id}.parquet"
        df.write_parquet(dataset_path)
        logger.info(f"Dataset saved to {dataset_path}")

    logger.info(f"Run {run_id} saved — {len(assignments)} codes across {n_clusters} clusters.")
    return run_id


# ---------------------------------------------------------------------------
# Build cluster profiles on demand
# ---------------------------------------------------------------------------


def build_cluster_profiles(
    run_id: int,
    df: pl.DataFrame,
    common_ids: list[int],
    fused_labels: np.ndarray,
) -> None:
    """Compute and persist cluster profiles for an existing run.

    Populates `cluster_profiles` with n_codes, total_cases, top_codes,
    temporal_signature and demographic_profile.

    LLM descriptions are NOT computed here — call populate_llm_descriptions()
    separately to avoid blocking the main pipeline.

    Args:
        run_id:       ID of an already-saved run.
        df:           Prepared dataframe for the run.
                      Required columns: terminology_id, code, code_type,
                      week, quantity, sex, age_group.
        common_ids:   Ordered list of terminology_ids matching fused_labels.
        fused_labels: 1-D numpy array of cluster labels.
    """
    id_to_cluster = {int(tid): int(label) for tid, label in zip(common_ids, fused_labels)}

    keys = pl.Series(list(id_to_cluster.keys()), dtype=pl.Int64)
    values = pl.Series(list(id_to_cluster.values()), dtype=pl.Int64)

    df_with_cluster = df.with_columns(
        pl.col("terminology_id").cast(pl.Int64).replace(keys, values).alias("cluster_id")
    ).filter(pl.col("cluster_id").is_not_null())

    for cluster_id in sorted(df_with_cluster["cluster_id"].unique().to_list()):
        cdf = df_with_cluster.filter(pl.col("cluster_id") == cluster_id)

        n_codes = cdf["terminology_id"].n_unique()
        total_cases = int(cdf["quantity"].sum())

        top_codes = (
            cdf.group_by(["terminology_id", "code", "code_type"])
            .agg(pl.col("quantity").sum().alias("total"))
            .sort("total", descending=True)
            .head(20)
            .to_dicts()
        )

        temporal_sig = (
            cdf.group_by("week")
            .agg(pl.col("quantity").sum().alias("total"))
            .sort("week")
            .to_dicts()
        )

        demo_profile = (
            cdf.with_columns((pl.col("sex") + "-" + pl.col("age_group")).alias("sex_age"))
            .group_by("sex_age")
            .agg(pl.col("quantity").sum().alias("total"))
            .with_columns((pl.col("total") / pl.col("total").sum()).alias("proportion"))
            .sort("sex_age")
            .to_dicts()
        )

        upsert_cluster_profile(
            run_id=run_id,
            cluster_id=cluster_id,
            n_codes=n_codes,
            total_cases=total_cases,
            top_codes=json.dumps(top_codes),
            temporal_signature=json.dumps(temporal_sig),
            demographic_profile=json.dumps(demo_profile),
        )

    logger.info(f"Profiles built for run {run_id}.")


# ---------------------------------------------------------------------------
# Explain clusters — per-view contribution + central code
# ---------------------------------------------------------------------------


def explain_clusters(
    run_id: int,
    common_ids: list[int],
    fused_labels: np.ndarray,
    clinical_distance: np.ndarray,
    demographic_distance: np.ndarray,
    temporal_distance: np.ndarray,
) -> None:
    """Compute and persist per-cluster interpretability signals.

    For each cluster, calculates:
    - Mean intra-cluster similarity per view (clinical, demographic, temporal)
    - Primary driver: which view has the highest intra-cluster similarity
    - Central code: the member with the lowest mean distance to all other members
      in the fused matrix (most representative code)

    Results are stored in cluster_profiles via upsert — safe to re-run.
    Requires build_cluster_profiles() to have been called first (profiles must exist).

    Args:
        run_id:               ID of an already-saved run.
        common_ids:           Ordered list of terminology_ids matching fused_labels.
        fused_labels:         1-D numpy array of cluster labels (same order as common_ids).
        clinical_distance:    Aligned clinical distance matrix (N x N, float32).
        demographic_distance: Aligned demographic distance matrix (N x N, float32).
        temporal_distance:    Aligned temporal distance matrix (N x N, float32).
    """
    from vir.db import fetch_cluster_profiles, get_connection

    profiles = fetch_cluster_profiles(run_id)
    if profiles.is_empty():
        logger.warning(f"No profiles found for run {run_id}. Run build_cluster_profiles first.")
        return

    # Convert distances to similarities: sim = 1 - normalised_distance
    def _to_similarity(dist: np.ndarray) -> np.ndarray:
        d = dist.copy().astype(np.float64)
        max_d = d.max()
        if max_d > 0:
            d = d / max_d
        return 1.0 - d

    clin_sim = _to_similarity(clinical_distance)
    demo_sim = _to_similarity(demographic_distance)
    temp_sim = _to_similarity(temporal_distance)

    # Build fused similarity from distances (1 - normalised)
    # We approximate the fused matrix via the mean of the three similarities
    # as a proxy for the SNF-fused matrix (which we don't persist)
    fused_sim_approx = (clin_sim + demo_sim + temp_sim) / 3.0

    id_to_idx = {int(tid): idx for idx, tid in enumerate(common_ids)}

    # Fetch code descriptions for central code annotation
    con = get_connection()
    code_meta = con.execute(
        "SELECT id, code, code_type, description FROM medical_terminologies"
    ).pl()
    id_to_meta = {
        row["id"]: {
            "code": row["code"],
            "code_type": row["code_type"],
            "description": row["description"],
        }
        for row in code_meta.iter_rows(named=True)
    }

    cluster_ids = sorted(set(int(label) for label in fused_labels))

    for cluster_id in cluster_ids:
        # Indices of members of this cluster
        member_indices = [
            id_to_idx[int(tid)]
            for tid, label in zip(common_ids, fused_labels)
            if int(label) == cluster_id and int(tid) in id_to_idx
        ]
        member_ids = [
            int(common_ids[i]) if isinstance(common_ids[0], (int, np.integer)) else common_ids[i]
            for i in member_indices
        ]

        if len(member_indices) < 2:
            continue

        ix = np.ix_(member_indices, member_indices)

        # Mean intra-cluster similarity per view (exclude diagonal)
        def _mean_sim(sim_matrix: np.ndarray) -> float:
            sub = sim_matrix[ix].copy()
            np.fill_diagonal(sub, np.nan)
            return float(np.nanmean(sub))

        clin_mean = _mean_sim(clin_sim)
        demo_mean = _mean_sim(demo_sim)
        temp_mean = _mean_sim(temp_sim)

        # Primary driver: view with highest intra-cluster similarity
        driver_map = {
            "clinical": clin_mean,
            "demographic": demo_mean,
            "temporal": temp_mean,
        }
        primary_driver = max(driver_map, key=lambda k: driver_map[k])

        # Central code: member with lowest mean distance to others in fused approx
        sub_fused = fused_sim_approx[ix].copy()
        np.fill_diagonal(sub_fused, np.nan)
        mean_sim_per_member = np.nanmean(sub_fused, axis=1)
        central_local_idx = int(np.argmax(mean_sim_per_member))
        central_tid = int(member_ids[central_local_idx])
        central_meta = id_to_meta.get(central_tid, {})

        # Per-member centrality list (descending), capped at 10 for LLM use
        sorted_member_indices = np.argsort(mean_sim_per_member)[::-1]
        member_centralities = [
            {
                "terminology_id": int(member_ids[local_idx]),
                "description": id_to_meta.get(int(member_ids[local_idx]), {}).get("description")
                or "",
                "centrality": round(float(mean_sim_per_member[local_idx]), 4),
            }
            for local_idx in sorted_member_indices
        ]

        explanation = {
            "view_similarity": {
                "clinical": round(clin_mean, 4),
                "demographic": round(demo_mean, 4),
                "temporal": round(temp_mean, 4),
            },
            "primary_driver": primary_driver,
            "central_code": {
                "terminology_id": central_tid,
                "code": central_meta.get("code"),
                "code_type": central_meta.get("code_type"),
                "description": central_meta.get("description"),
            },
            "member_centralities": member_centralities,
        }

        upsert_cluster_profile(
            run_id=run_id,
            cluster_id=cluster_id,
            explanation=json.dumps(explanation),
        )

    logger.info(f"Explanations computed for {len(cluster_ids)} clusters in run {run_id}.")


# ---------------------------------------------------------------------------
# Populate LLM descriptions — on demand, separate step
# ---------------------------------------------------------------------------


def populate_llm_descriptions(run_id: int) -> None:
    """Generate and store LLM descriptions for all clusters in a run.

    Reads top_codes from cluster_profiles and calls describe_cluster.
    Safe to re-run — already-described clusters are skipped.
    """
    from vir.db import fetch_cluster_profiles
    from vir.modeling.clustering.describe import describe_cluster

    profiles = fetch_cluster_profiles(run_id)
    if profiles.is_empty():
        logger.warning(f"No profiles for run {run_id}. Run build_cluster_profiles first.")
        return

    for row in profiles.iter_rows(named=True):
        if row["llm_short_description"]:
            continue
        if not row["top_codes"]:
            continue

        explanation = json.loads(row["explanation"]) if row.get("explanation") else {}
        member_centralities = explanation.get("member_centralities") if explanation else None

        if member_centralities:
            labels_count = {
                m["description"]: m["centrality"]
                for m in member_centralities[:10]
                if m.get("description")
            }
        else:
            top_codes = json.loads(row["top_codes"])
            labels_count = {f"{c['code']} ({c['code_type']})": c["total"] for c in top_codes[:10]}

        desc = describe_cluster(labels_count)

        upsert_cluster_profile(
            run_id=run_id,
            cluster_id=row["cluster_id"],
            llm_short_description=desc.get("short_description"),
            llm_description=desc.get("description"),
            llm_abbreviation=desc.get("abbreviation"),
        )

    logger.info(f"LLM descriptions populated for run {run_id}.")
