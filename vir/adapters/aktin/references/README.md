# AKTIN adapter references

## icd10gm2025syst_kodes.txt

Source: BfArM (Bundesinstitut für Arzneimittel und Medizinprodukte), ICD-10-GM
Version 2025 — Systematisches Verzeichnis (classification file
`icd10gm2025syst-meta.zip` → `Klassifikationsdateien/icd10gm2025syst_kodes.txt`).

Stand der Klassifikation: 2024-09-13. OID `1.2.276.0.76.5.548`.

Format: semicolon-separated, no header, one row per code. ~16,800 rows (4 MB).

Redistribution / attribution: subject to BfArM download conditions
(`downloadbedingungen-2024.pdf`). License terms must be confirmed before
making the public repo containing this file generally available — tracked
in `AKTIN-PLAN.md § Stakeholder check-ins`.

Parsing of this file lives in `vir/adapters/aktin/terminology.py`.
