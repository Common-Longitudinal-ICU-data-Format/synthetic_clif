"""CLIF 2.1 -> 3.0 value conventions for standardized columns.

CLIF 3.0 lowercases and snake_cases every ``*_category`` / ``*_group`` /
``*_type`` value (``&`` -> ``_and_``, ``/`` -> ``_or_``) and renames a small
set of values outright. The renames below follow the official crosswalk shipped
with clifpy (``schemas/crosswalks/clif_2.1_to_3.0.yaml``), plus the values this
generator emitted in 2.1 that have no automatic 3.0 equivalent.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

import pandas as pd

from synthetic_clif.clif3 import schema

# Official non-derivable renames (clifpy crosswalk, CLIF 2.1 -> 3.0)
OFFICIAL_RENAMES: dict[str, dict[str, dict[str, str]]] = {
    "patient": {"language_category": {"Unknown or NA": "unknown"}},
    "hospitalization": {
        "discharge_category": {
            "Skilled Nursing Facility (SNF)": "snf",
            "Acute Inpatient Rehab Facility": "acute_ip_rehab",
            "Long Term Care Hospital (LTACH)": "ltach",
            "Acute Care Hospital": "acute_care_hosp",
            "Against Medical Advice (AMA)": "ama",
            "Psychiatric Hospital": "mental_health_hosp",
        }
    },
    "labs": {
        "lab_category": {"glucose_fingerstick": "glucose", "glucose_serum": "glucose"},
        "lab_specimen_category": {"blood/plasma/serum": "plasma_blood"},
    },
    "respiratory_support": {
        "device_category": {"High Flow NC": "hfnc"},
        "mode_category": {
            "Assist Control-Volume Control": "acvc",
            "Pressure-Regulated Volume Control": "prvc",
            "Pressure Support/CPAP": "ps_or_cpap",
        },
    },
    "microbiology_culture": {
        "organism_category": {
            "clostridium_difficile": "clostridioides_difficile",
            "staphyloccocus_coagneg": "staphylococcus_coagneg",
        }
    },
}

# Values synthetic_clif emitted in 2.1 that need a hand-chosen 3.0 value
GENERATOR_RENAMES: dict[str, dict[str, dict[str, str]]] = {
    "labs": {"lab_specimen_category": {"blood": "plasma_blood"}},
    # 3.0 has a single albumin split by concentration; continuous albumin
    # infusions here are volume expansion, i.e. 5% albumin.
    "medication_admin_continuous": {"med_category": {"albumin": "albumin_5"}},
    "medication_admin_intermittent": {"med_category": {"albumin": "albumin_5"}},
    "medication_orders": {"med_category": {"albumin": "albumin_5"}},
    "patient_diagnosis": {
        "diagnosis_code_format": {"ICD-10-CM": "ICD10CM", "icd10cm": "ICD10CM"},
        "source_type": {
            "Encounter": "encounter_dx",
            "Problem List": "problem_list",
            "Medical History": "medical_history",
        },
    },
    "consult_orders": {
        "order_category": {
            "PT Evaluation": "pt",
            "OT Evaluation": "ot",
            "Speech Evaluation": "speech",
            "Nutrition Consult": "nutrition",
            "Social Work Consult": "social_work",
            "Palliative Care Consult": "palliative_care",
        },
        "order_status_category": {
            "Completed": "completed",
            "Active": "sent",
            "Discontinued": "canceled",
        },
    },
    "transfusion": {
        "component_category": {
            "Packed RBCs": "prbc",
            "Fresh Frozen Plasma": "plasma",
            "Platelets": "platelets",
            "Cryoprecipitate": "cryoprecipitate",
            "Whole Blood": "whole_blood",
        }
    },
    "invasive_hemodynamics": {
        "measurement_category": {
            "CVP": "cvp",
            "PA Systolic": "pa_systolic",
            "PA Diastolic": "pa_diastolic",
            "PA Mean": "pa_mean",
            "PCWP": "pcwp",
            "Cardiac Output": "co_thermodilution",
            "Cardiac Index": "co_index",
            "SVR": "svr",
            # PVR and SvO2 have no 3.0 category (SvO2 is a lab: so2_mixed_venous)
        }
    },
}


def normalize_category_value(value) -> Optional[str]:
    """Apply the CLIF 3.0 value convention: lowercase snake_case.

    ``&`` -> ``_and_`` and ``/`` -> ``_or_``; other punctuation and whitespace
    collapse to single underscores.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value is pd.NA:
        return None
    s = str(value).strip()
    s = s.replace("&", "_and_").replace("/", "_or_")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def _rename_for(table: str, col: str) -> dict[str, str]:
    out: dict[str, str] = {}
    out.update(OFFICIAL_RENAMES.get(table, {}).get(col, {}))
    out.update(GENERATOR_RENAMES.get(table, {}).get(col, {}))
    return out


def to_v3_value(table: str, col: str, value, permissible: Optional[Iterable] = None):
    """Map one 2.1-style value to its 3.0 value (unchanged if already valid)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return value
    pv = set(map(str, permissible)) if permissible is not None else None
    v = str(value)
    if pv is not None and v in pv:
        return v
    renames = _rename_for(table, col)
    if v in renames:
        return renames[v]
    if pv is not None and normalize_category_value(v) in pv:
        return normalize_category_value(v)
    # Some permissible lists are not snake_case (e.g. ICD10CM); try case-insensitive
    if pv is not None:
        lower = {p.lower(): p for p in pv}
        if v.lower() in lower:
            return lower[v.lower()]
    return normalize_category_value(v) if pv is None else v


def map_column(df: pd.DataFrame, table: str, col: str) -> pd.Series:
    """Vectorized ``to_v3_value`` over one column (maps unique values once)."""
    pv = schema.permissible_values(table, col)
    s = df[col]
    uniques = pd.unique(s.dropna())
    mapping = {u: to_v3_value(table, col, u, pv) for u in uniques}
    return s.map(lambda x: mapping.get(x, x) if not _isna(x) else x)


def _isna(x) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False
