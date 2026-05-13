"""AKTIN-specific colour palettes for plotly visualisations."""

from vir.viz import TABLEAU_COLORS

SEX_COLOR: dict[str, str] = {
    "M": TABLEAU_COLORS[0],
    "F": TABLEAU_COLORS[2],
    "X": TABLEAU_COLORS[7],
}

AGE_GROUP_COLOR: dict[str, str] = {
    "0-4": TABLEAU_COLORS[4],
    "5-9": TABLEAU_COLORS[8],
    "10-14": TABLEAU_COLORS[3],
    "15-19": TABLEAU_COLORS[9],
    "20-39": TABLEAU_COLORS[1],
    "40-59": TABLEAU_COLORS[5],
    "60-79": TABLEAU_COLORS[6],
    "80+": TABLEAU_COLORS[2],
}

CODE_TYPE_COLOR: dict[str, str] = {
    "icd10gm": TABLEAU_COLORS[0],
}
