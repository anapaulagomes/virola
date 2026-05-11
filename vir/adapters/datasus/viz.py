"""DATASUS-specific colour palettes for plotly visualisations.

These map the DATASUS-specific code types, sex categories, and reduced age
buckets to colours used across the DATASUS notebooks (cross-city, viz_*,
view_demographic).
"""

import plotly.express as px

from vir.viz import TABLEAU_COLORS

CODE_TYPE_COLOR: dict[str, str] = {
    "CID10": px.colors.qualitative.Plotly[0],
    "CIAP2": px.colors.qualitative.Plotly[1],
    "AB": px.colors.qualitative.Plotly[2],
}

SEX_COLOR: dict[str, str] = {
    "Masculino": TABLEAU_COLORS[0],
    "Feminino": TABLEAU_COLORS[2],
    "Sem Informação": TABLEAU_COLORS[5],
}

AGE_GROUP_COLOR: dict[str, str] = {
    "< 1 ano": TABLEAU_COLORS[4],
    "1 a 15 anos": TABLEAU_COLORS[3],
    "16 a 59 anos": TABLEAU_COLORS[1],
    "Maior que 60 anos": TABLEAU_COLORS[6],
}
