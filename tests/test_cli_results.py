from typer.testing import CliRunner

import vir.db as db_module
from vir.cli import app

runner = CliRunner()


def _seed_run(city: str | None, dataset_name: str | None = None) -> int:
    db_module.create_embeddings_table_if_not_exists()
    return db_module.insert_run(
        dataset_name=dataset_name or f"dataset_{city or 'unknown'}",
        city=city,
        region="Norte",
        population_range="> 500 mil",
        year_min=2022,
        year_max=2024,
        snf_k=10,
        n_clusters=2,
    )


class TestResultsList:
    def test_empty_db_emits_message(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        result = runner.invoke(app, ["results", "list"])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_shows_more_rows_than_polars_default(self, tmp_db):
        for index in range(12):
            _seed_run("Manaus", dataset_name=f"dx{index:02d}")
        result = runner.invoke(app, ["results", "list"])
        assert result.exit_code == 0
        for index in range(12):
            assert f"dx{index:02d}" in result.output

    def test_limit_caps_to_n_most_recent(self, tmp_db):
        for index in range(5):
            _seed_run("Manaus", dataset_name=f"dx{index:02d}")
        result = runner.invoke(app, ["results", "list", "--limit", "2"])
        assert result.exit_code == 0
        assert "dx04" in result.output
        assert "dx03" in result.output
        assert "dx02" not in result.output
        assert "dx00" not in result.output


class TestResultsDelete:
    def test_errors_when_neither_id_nor_city_given(self, tmp_db):
        result = runner.invoke(app, ["results", "delete"])
        assert result.exit_code != 0
        assert "--id" in result.output
        assert "--city" in result.output

    def test_errors_when_both_id_and_city_given(self, tmp_db):
        result = runner.invoke(app, ["results", "delete", "--id", "1", "--city", "Manaus"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_delete_by_id_with_yes_skips_confirmation(self, tmp_db):
        run_id = _seed_run("Manaus")
        result = runner.invoke(app, ["results", "delete", "--id", str(run_id), "--yes"])
        assert result.exit_code == 0
        assert db_module.fetch_runs().is_empty()

    def test_delete_by_id_aborts_on_n(self, tmp_db):
        run_id = _seed_run("Manaus")
        result = runner.invoke(app, ["results", "delete", "--id", str(run_id)], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert not db_module.fetch_runs().is_empty()

    def test_delete_by_id_proceeds_on_y(self, tmp_db):
        run_id = _seed_run("Manaus")
        result = runner.invoke(app, ["results", "delete", "--id", str(run_id)], input="y\n")
        assert result.exit_code == 0
        assert db_module.fetch_runs().is_empty()

    def test_delete_by_id_unknown_emits_message(self, tmp_db):
        db_module.create_embeddings_table_if_not_exists()
        db_module.create_results_schema()
        result = runner.invoke(app, ["results", "delete", "--id", "9999", "--yes"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_delete_by_city_lists_and_deletes_all(self, tmp_db):
        manaus_a = _seed_run("Manaus")
        manaus_b = _seed_run("Manaus")
        bauru = _seed_run("Bauru")

        result = runner.invoke(app, ["results", "delete", "--city", "Manaus", "--yes"])

        assert result.exit_code == 0
        remaining_ids = db_module.fetch_runs()["run_id"].to_list()
        assert remaining_ids == [bauru]
        assert manaus_a not in remaining_ids
        assert manaus_b not in remaining_ids

    def test_delete_by_city_no_matches_emits_message(self, tmp_db):
        _seed_run("Bauru")
        result = runner.invoke(app, ["results", "delete", "--city", "Manaus", "--yes"])
        assert result.exit_code == 0
        assert "No runs to delete" in result.output

    def test_delete_by_city_ignores_null_city(self, tmp_db):
        unmapped = _seed_run(None)
        bauru = _seed_run("Bauru")

        result = runner.invoke(app, ["results", "delete", "--city", "Bauru", "--yes"])

        assert result.exit_code == 0
        remaining_ids = db_module.fetch_runs()["run_id"].to_list()
        assert remaining_ids == [unmapped]
        assert bauru not in remaining_ids

    def test_invalid_city_name_rejected(self, tmp_db):
        result = runner.invoke(app, ["results", "delete", "--city", "Atlantis", "--yes"])
        assert result.exit_code != 0
        assert "Invalid city" in result.output
