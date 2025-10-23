from vir.modeling.clustering.describe import describe_cluster


class TestDescribeCluster:
    def test_describe(self):
        terminology_labels_count = {
            "Fever": 100,
            "Pain attributed to respiratory tract": 67,
            "Difficulty breathing, dyspnea": 55,
            "Noisy breathing": 17,
        }
        {
            "abbreviation": "RTIS",
            "description": "A syndrome characterized by fever, pain attributed to the respiratory tract, ...tion within the syndromic surveillance system.",
            "short_description": "Respiratory Tract Inflammation Syndrome (RTIS)",
        }
        description = describe_cluster(terminology_labels_count)
        assert list(description.keys()) == ["short_description", "description", "abbreviation"]
