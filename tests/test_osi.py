import json
import re

import pytest

from vir.osi import (
    CODE_SYSTEM_BY_CODE_TYPE,
    OSI_DATETIME_FORMAT,
    OSI_TEMPLATE_PATH,
    build_inclusion_criterion,
    build_osi_payload,
    load_osi_template,
    osi_payload_as_json,
)

OSI_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class TestLoadOsiTemplate:
    def test_loads_default_template_from_references(self):
        template = load_osi_template()
        assert template["definition_type"] == "syndrome_definition"
        assert template["open_syndrome_version"] == "1.0.0"
        assert template["inclusion_criteria"] == []

    def test_returns_independent_copies(self):
        first = load_osi_template()
        first["title"] = "mutated"
        second = load_osi_template()
        assert second["title"] == ""

    def test_accepts_explicit_template_path(self, tmp_path):
        template_path = tmp_path / "custom_template.json"
        template_path.write_text(json.dumps({"title": "custom", "inclusion_criteria": []}))
        template = load_osi_template(template_path)
        assert template == {"title": "custom", "inclusion_criteria": []}

    def test_default_template_path_points_to_references(self):
        assert OSI_TEMPLATE_PATH.exists()
        assert OSI_TEMPLATE_PATH.name == "osi_template.json"


class TestBuildInclusionCriterion:
    def test_maps_cid10_to_icd10(self):
        criterion = build_inclusion_criterion(
            code="A692", code_type="cid10", description="Lyme disease"
        )
        assert criterion == {
            "type": "diagnosis",
            "name": "Lyme disease",
            "code": {"system": "CID-10", "code": "A692"},
        }

    def test_maps_ciap2_to_icpc2(self):
        criterion = build_inclusion_criterion(
            code="R74", code_type="ciap2", description="Infecção respiratória aguda"
        )
        assert criterion["code"] == {"system": "CIAP-2", "code": "R74"}

    def test_maps_ab_code_type(self):
        criterion = build_inclusion_criterion(
            code="AB123", code_type="ab", description="Some AB programme code"
        )
        assert criterion["code"]["system"] == "AB"

    def test_code_type_is_case_insensitive(self):
        criterion = build_inclusion_criterion(
            code="A692", code_type="CID10", description="Lyme disease"
        )
        assert criterion["code"]["system"] == "CID-10"

    def test_unknown_code_type_falls_back_to_uppercase(self):
        criterion = build_inclusion_criterion(
            code="X1", code_type="snomed", description="Whatever"
        )
        assert criterion["code"]["system"] == "SNOMED"

    def test_known_systems_match_constant(self):
        assert CODE_SYSTEM_BY_CODE_TYPE == {
            "cid10": "CID-10",
            "ciap2": "CIAP-2",
            "ab": "AB",
        }


class TestBuildOsiPayload:
    @pytest.fixture
    def sample_members(self):
        return [
            {"code": "A692", "code_type": "cid10", "description": "Lyme disease"},
            {"code": "R74", "code_type": "ciap2", "description": "Infecção respiratória"},
        ]

    def test_inclusion_criteria_lists_one_entry_per_member(self, sample_members):
        payload = build_osi_payload(sample_members, title="Respiratory cluster")
        assert len(payload["inclusion_criteria"]) == 2
        assert payload["inclusion_criteria"][0] == {
            "type": "diagnosis",
            "name": "Lyme disease",
            "code": {"system": "CID-10", "code": "A692"},
        }
        assert payload["inclusion_criteria"][1]["code"]["system"] == "CIAP-2"

    def test_uses_provided_title_and_description(self, sample_members):
        payload = build_osi_payload(
            sample_members, title="Respiratory cluster", description="A draft syndrome"
        )
        assert payload["title"] == "Respiratory cluster"
        assert payload["description"] == "A draft syndrome"

    def test_published_at_defaults_to_utc_now_in_osi_format(self, sample_members):
        payload = build_osi_payload(sample_members, title="cluster")
        assert OSI_DATETIME_PATTERN.match(payload["published_at"])

    def test_published_at_can_be_overridden(self, sample_members):
        payload = build_osi_payload(
            sample_members, title="cluster", published_at="2020-05-30T12:00:00Z"
        )
        assert payload["published_at"] == "2020-05-30T12:00:00Z"

    def test_osi_datetime_format_constant_matches_example(self):
        assert OSI_DATETIME_FORMAT == "%Y-%m-%dT%H:%M:%SZ"

    def test_preserves_template_envelope_fields(self, sample_members):
        payload = build_osi_payload(sample_members, title="cluster")
        assert payload["scope"] == "broad"
        assert payload["category"] == "suspected"
        assert payload["definition_type"] == "syndrome_definition"
        assert payload["organization"] == "Fiocruz/Cidacs Bahia"
        assert payload["language"] == "Portuguese"
        assert payload["location"] == "Brazil"

    def test_does_not_mutate_template_on_disk(self, sample_members):
        build_osi_payload(sample_members, title="first call")
        fresh_template = load_osi_template()
        assert fresh_template["title"] == ""
        assert fresh_template["inclusion_criteria"] == []

    def test_accepts_iterators(self, sample_members):
        payload = build_osi_payload(iter(sample_members), title="cluster")
        assert len(payload["inclusion_criteria"]) == 2

    def test_empty_member_list_yields_empty_inclusion_criteria(self):
        payload = build_osi_payload([], title="empty cluster")
        assert payload["inclusion_criteria"] == []
        assert payload["title"] == "empty cluster"

    def test_member_extra_keys_are_ignored(self):
        members = [
            {
                "code": "A692",
                "code_type": "cid10",
                "description": "Lyme disease",
                "total_cases": 42,
                "centrality": 0.81,
            }
        ]
        payload = build_osi_payload(members, title="cluster")
        assert payload["inclusion_criteria"][0] == {
            "type": "diagnosis",
            "name": "Lyme disease",
            "code": {"system": "CID-10", "code": "A692"},
        }


class TestOsiPayloadAsJson:
    def test_serialises_to_indented_unicode_json(self):
        payload = build_osi_payload(
            [{"code": "A692", "code_type": "cid10", "description": "Doença de Lyme"}],
            title="cluster",
        )
        serialised = osi_payload_as_json(payload)
        assert "Doença de Lyme" in serialised
        assert json.loads(serialised) == payload

    def test_indent_is_configurable(self):
        payload = build_osi_payload([], title="cluster")
        compact = osi_payload_as_json(payload, indent=0)
        assert "\n" in compact
        assert json.loads(compact) == payload
