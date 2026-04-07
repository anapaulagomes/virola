import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from vir.config import REFERENCES_DIR

OSI_TEMPLATE_PATH = REFERENCES_DIR / "osi_template.json"
OSI_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _now_utc_iso() -> str:
    """Return the current UTC time formatted as ``2020-05-30T12:00:00Z``."""
    return datetime.now(timezone.utc).strftime(OSI_DATETIME_FORMAT)


CODE_SYSTEM_BY_CODE_TYPE = {
    "cid10": "CID-10",
    "ciap2": "CIAP-2",
    "ab": "AB",
}


def load_osi_template(template_path: Path | None = None) -> dict:
    """Return a deep copy of the OSI envelope template as a dictionary."""
    path = template_path or OSI_TEMPLATE_PATH
    with path.open(encoding="utf-8") as template_file:
        return json.load(template_file)


def build_inclusion_criterion(code: str, code_type: str, description: str) -> dict:
    """Build a single OSI ``inclusion_criteria`` entry for a diagnosis code.

    Each entry follows the OSI ``diagnosis`` shape::

        {
            "type": "diagnosis",
            "name": "Lyme disease",
            "code": {"system": "ICD-10", "code": "A692"}
        }

    Unknown ``code_type`` values fall back to their uppercase representation
    so the payload still validates structurally.
    """
    code_system = CODE_SYSTEM_BY_CODE_TYPE.get(code_type.lower(), code_type.upper())
    return {
        "type": "diagnosis",
        "name": description,
        "code": {
            "system": code_system,
            "code": code,
        },
    }


def build_osi_payload(
    members: Iterable[dict],
    *,
    title: str,
    description: str = "",
    published_at: str | None = None,
    template_path: Path | None = None,
) -> dict:
    """Return an OSI payload populated from a list of selected member codes.

    Args:
        members: Iterable of dicts with at least ``code``, ``code_type`` and
            ``description`` keys (extra keys are ignored).
        title: Human-readable title for the syndrome candidate.
        description: Optional long-form description.
        published_at: ISO datetime string in the OSI format
            (``YYYY-MM-DDTHH:MM:SSZ``). Defaults to the current UTC time.
        template_path: Optional override for the OSI envelope path; useful in
            tests.

    Returns:
        A dictionary that mirrors ``references/osi_template.json`` with the
        ``title``, ``description``, ``published_at`` and ``inclusion_criteria``
        fields filled in.
    """
    payload = load_osi_template(template_path)
    payload["title"] = title
    payload["description"] = description
    payload["published_at"] = published_at or _now_utc_iso()
    payload["inclusion_criteria"] = [
        build_inclusion_criterion(
            code=member["code"],
            code_type=member["code_type"],
            description=member["description"],
        )
        for member in members
    ]
    return copy.deepcopy(payload)


def osi_payload_as_json(payload: dict, *, indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=False)
