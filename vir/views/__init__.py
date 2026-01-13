"""
Views module for multi-view clustering.

Each view computes a distance/similarity matrix for diagnoses based on different perspectives:
- Temporal: Time series similarity (when)
- Demographic: Age/sex distribution similarity (who)
- Clinical: Semantic/ontological similarity (what)
"""

from vir.views.base import BaseView
from vir.views.clinical import ClinicalView
from vir.views.demographic import DemographicView
from vir.views.temporal import TemporalView

__all__ = [
    "BaseView",
    "ClinicalView",
    "DemographicView",
    "TemporalView",
]
