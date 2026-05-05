import json
from unittest.mock import patch

from vir.clusters import describe_cluster


class TestDescribeCluster:
    @patch("vir.clusters.descriptions_cache")
    @patch("vir.clusters.llm")
    def test_describe(self, mock_llm, mock_cache):
        mock_cache.get.return_value = None
        mock_llm.get_model.return_value.prompt.return_value.text.return_value = json.dumps(
            {
                "abbreviation": "RTIS",
                "description": "A syndrome characterized by fever, pain attributed to the respiratory tract, ...tion within the syndromic surveillance system.",
                "short_description": "Respiratory Tract Inflammation Syndrome (RTIS)",
            }
        )
        terminology_labels_count = {
            "Fever": 100,
            "Pain attributed to respiratory tract": 67,
            "Difficulty breathing, dyspnea": 55,
            "Noisy breathing": 17,
        }
        description = describe_cluster(terminology_labels_count)
        assert sorted(list(description.keys())) == sorted(
            ["short_description", "description", "abbreviation"]
        )
        assert len(mock_llm.mock_calls) == 3
        assert len(mock_cache.mock_calls) == 2

    @patch("vir.clusters.descriptions_cache")
    @patch("vir.clusters.llm")
    def test_cache_describe_cluster(self, mock_llm, mock_cache):
        cluster_description = json.dumps(
            {
                "abbreviation": "RTIS",
                "description": "A syndrome characterized by fever, pain attributed to the respiratory tract, ...tion within the syndromic surveillance system.",
                "short_description": "Respiratory Tract Inflammation Syndrome (RTIS)",
            }
        )
        mock_llm.get_model.return_value.prompt.return_value.text.return_value = cluster_description
        mock_cache.get.return_value = None
        terminology_labels_count = {
            "Fever": 100,
            "Pain attributed to respiratory tract": 67,
            "Difficulty breathing, dyspnea": 55,
            "Noisy breathing": 17,
        }
        describe_cluster(terminology_labels_count)
        assert len(mock_llm.mock_calls) == 3
        assert len(mock_cache.mock_calls) == 2

        mock_cache.get.return_value = cluster_description
        describe_cluster(terminology_labels_count)
        assert len(mock_llm.mock_calls) == 3
        assert len(mock_cache.mock_calls) == 4
