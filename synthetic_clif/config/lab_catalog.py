"""Lab categories the generator produces, with their CLIF 2.1.0 and 3.0.0 metadata.

One entry per ``lab_category``. Units and order categories come from the
official mCIDE files (``mCIDE/labs/clif_lab_categories.csv`` on the CLIF
``main`` branch for 2.1.0 and on the ``3.0`` branch for 3.0.0). LOINC codes
are for the blood/serum/plasma test the generator simulates.

The 52 CLIF 2.1.0 categories are all generated. ``CLIF3_ONLY`` lists the extra
3.0.0 categories (not part of the 2.1.0 vocabulary); they appear only in 3.0.0
output. In 3.0.0, ``glucose_serum`` and ``glucose_fingerstick`` both become
``glucose`` (told apart by ``lab_order_category``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LabSpec:
    unit_21: Optional[str]          # CLIF 2.1.0 reference_unit (None: not a 2.1 category)
    unit_30: str                    # CLIF 3.0.0 reference_unit
    order_21: Optional[str]         # CLIF 2.1.0 lab_order_category
    order_30: Optional[str]         # CLIF 3.0.0 lab_order_category (None: keep the 2.1 value)
    loinc: str
    decimals: int
    # Fallback draw when no course is available: mean, sd, lower, upper clip
    params: tuple


_P = LabSpec

LABS: dict[str, LabSpec] = {
    # Basic metabolic panel
    "sodium": _P("mmol/L", "mmol/l", "bmp", "bmp", "2951-2", 1, (140, 3, 110, 175)),
    "potassium": _P("mmol/L", "mmol/l", "bmp", "bmp", "2823-3", 1, (4.0, 0.5, 2.0, 8.0)),
    "chloride": _P("mmol/L", "mmol/l", "bmp", "bmp", "2075-0", 1, (102, 3, 75, 135)),
    # bicarbonate is reported on BMPs and blood gases; 3.0 keeps whichever
    "bicarbonate": _P("mmol/L", "mmol/l", "bmp", None, "1963-8", 1, (24, 3, 5, 45)),
    "bun": _P("mg/dL", "mg/dl", "bmp", "bmp", "3094-0", 1, (15, 8, 2, 200)),
    "creatinine": _P("mg/dL", "mg/dl", "bmp", "bmp", "2160-0", 1, (1.0, 0.5, 0.2, 15.0)),
    "glucose_serum": _P("mg/dL", "mg/dl", "bmp", "bmp", "2345-7", 1, (110, 40, 20, 900)),
    "glucose_fingerstick": _P("mg/dL", "mg/dl", "misc", "poc", "41653-7", 0, (130, 40, 20, 600)),
    "calcium_total": _P("mg/dL", "mg/dl", "bmp", "bmp", "17861-6", 1, (9.0, 0.6, 5.0, 15.0)),
    "calcium_ionized": _P("mg/dL", "mg/dl", "misc", "misc", "17863-2", 2, (4.8, 0.3, 2.5, 7.0)),
    "magnesium": _P("mg/dL", "mg/dl", "misc", "misc", "19123-9", 1, (2.0, 0.3, 0.8, 5.0)),
    "phosphate": _P("mg/dL", "mg/dl", "misc", "misc", "2777-1", 1, (3.5, 0.8, 0.8, 12.0)),
    # Liver panel
    "albumin": _P("g/dL", "g/dl", "lft", "lft", "1751-7", 1, (3.5, 0.6, 1.0, 5.5)),
    "total_protein": _P("g/dL", "g/dl", "lft", "lft", "2885-2", 1, (7.0, 0.8, 3.0, 11.0)),
    "ast": _P("U/L", "u/l", "lft", "lft", "1920-8", 0, (30, 20, 5, 10000)),
    "alt": _P("U/L", "u/l", "lft", "lft", "1742-6", 0, (30, 20, 3, 8000)),
    "alkaline_phosphatase": _P("U/L", "u/l", "lft", "lft", "6768-6", 0, (80, 30, 20, 2000)),
    "bilirubin_total": _P("mg/dL", "mg/dl", "lft", "lft", "1975-2", 1, (0.8, 0.5, 0.1, 40.0)),
    "bilirubin_conjugated": _P("mg/dL", "mg/dl", "lft", "lft", "1968-7", 1, (0.2, 0.2, 0.0, 30.0)),
    "bilirubin_unconjugated": _P("mg/dL", "mg/dl", "lft", "lft", "1971-1", 1, (0.5, 0.3, 0.0, 20.0)),
    "ldh": _P("U/L", "u/l", "misc", "misc", "2532-0", 0, (200, 60, 80, 8000)),
    # Complete blood count with differential
    "hemoglobin": _P("g/dL", "g/dl", "cbc", "cbc", "718-7", 1, (12, 2, 3, 20)),
    "wbc": _P("10^3/µL", "10^3/ul", "cbc", "cbc", "6690-2", 1, (8, 4, 0.1, 150)),
    "platelet_count": _P("10^3/µL", "10^3/ul", "cbc", "cbc", "777-3", 0, (220, 80, 2, 1200)),
    "neutrophils_percent": _P("%", "%", "cbc", "cbc", "770-8", 1, (62, 8, 0, 100)),
    "lymphocytes_percent": _P("%", "%", "cbc", "cbc", "736-9", 1, (26, 7, 0, 100)),
    "monocytes_percent": _P("%", "%", "cbc", "cbc", "5905-5", 1, (7, 2, 0, 100)),
    "eosinophils_percent": _P("%", "%", "cbc", "cbc", "713-8", 1, (2.5, 1.5, 0, 100)),
    "basophils_percent": _P("%", "%", "cbc", "cbc", "706-2", 1, (0.6, 0.3, 0, 100)),
    "neutrophils_absolute": _P("10^3/µL", "10^3/ul", "cbc", "cbc", "751-8", 2, (5.0, 3.0, 0, 120)),
    # the 2.1.0 mCIDE files absolute lymphocytes under misc; 3.0.0 moves them to cbc
    "lymphocytes_absolute": _P("10^3/µL", "10^3/ul", "misc", "cbc", "731-0", 2, (2.0, 0.8, 0, 50)),
    "monocytes_absolute": _P("10^3/µL", "10^3/ul", "cbc", "cbc", "742-7", 2, (0.5, 0.2, 0, 20)),
    "eosinophils_absolute": _P("10^3/µL", "10^3/ul", "cbc", "cbc", "711-2", 2, (0.2, 0.1, 0, 10)),
    "basophils_absolute": _P("10^3/µL", "10^3/ul", "cbc", "cbc", "704-7", 2, (0.05, 0.03, 0, 5)),
    # Coagulation
    "inr": _P("(no units)", "(no units)", "coags", "coags", "6301-6", 1, (1.1, 0.3, 0.8, 12.0)),
    "pt": _P("sec", "sec", "coags", "coags", "5902-2", 1, (12, 2, 8, 120)),
    "ptt": _P("sec", "sec", "coags", "coags", "3173-2", 1, (30, 5, 18, 200)),
    # Blood gases
    "ph_arterial": _P("(no units)", "(no units)", "blood_gas", "blood_gas", "2744-1", 2, (7.40, 0.05, 6.7, 7.8)),
    "pco2_arterial": _P("mmHg", "mmhg", "blood_gas", "blood_gas", "2019-8", 1, (40, 5, 12, 140)),
    "po2_arterial": _P("mmHg", "mmhg", "blood_gas", "blood_gas", "2703-7", 1, (85, 15, 25, 600)),
    "so2_arterial": _P("%", "%", "blood_gas", "blood_gas", "2708-6", 1, (96, 2, 40, 100)),
    "ph_venous": _P("(no units)", "(no units)", "blood_gas", "blood_gas", "2746-6", 2, (7.36, 0.05, 6.7, 7.8)),
    "pco2_venous": _P("mmHg", "mmhg", "blood_gas", "blood_gas", "2021-4", 1, (46, 5, 15, 150)),
    "so2_central_venous": _P("%", "%", "blood_gas", "blood_gas", "97549-0", 1, (70, 8, 20, 98)),
    "so2_mixed_venous": _P("%", "%", "blood_gas", "mixed_venous_blood_gas", "19224-5", 1, (68, 8, 20, 95)),
    "lactate": _P("mmol/L", "mmol/l", "misc", "misc", "2524-7", 1, (1.5, 1.0, 0.2, 30.0)),
    # Cardiac and inflammatory markers
    "troponin_i": _P("ng/L", "ng/l", "misc", "cardiac", "89579-7", 1, (8, 6, 0, 100000)),
    "troponin_t": _P("ng/L", "ng/l", "misc", "cardiac", "67151-1", 1, (12, 8, 0, 100000)),
    "crp": _P("mg/L", "mg/l", "misc", "inflammatory", "1988-5", 1, (5, 10, 0, 500)),
    "procalcitonin": _P("ng/mL", "ng/ml", "misc", "inflammatory", "33959-8", 2, (0.3, 0.5, 0, 500)),
    "esr": _P("mm/hour", "mm/hour", "misc", "inflammatory", "4537-7", 0, (20, 12, 0, 140)),
    "ferritin": _P("ng/mL", "ng/ml", "misc", "inflammatory", "2276-4", 0, (150, 100, 5, 100000)),
    # CLIF 3.0.0 only
    "hematocrit": _P(None, "%", None, "cbc", "4544-3", 1, (36, 6, 10, 65)),
    "fibrinogen": _P(None, "mg/dl", None, "coags", "3255-7", 0, (300, 80, 30, 1200)),
    "d_dimer": _P(None, "ug/ml", None, "coags", "48065-7", 2, (0.5, 0.4, 0.1, 80.0)),
    "po2_mixed_venous": _P(None, "mmhg", None, "mixed_venous_blood_gas", "19211-2", 1, (38, 4, 15, 80)),
    "bnp": _P(None, "pg/ml", None, "cardiac", "30934-4", 0, (80, 60, 5, 10000)),
    "nt_pro_bnp": _P(None, "pg/ml", None, "cardiac", "33762-6", 0, (300, 250, 10, 70000)),
    "triglycerides": _P(None, "mg/dl", None, "misc", "2571-8", 0, (130, 50, 20, 3000)),
}

CLIF21_LAB_CATEGORIES: list[str] = [k for k, v in LABS.items() if v.unit_21 is not None]
CLIF3_ONLY: frozenset[str] = frozenset(k for k, v in LABS.items() if v.unit_21 is None)

# 3.0 renames (applied by synthetic_clif.clif3.vocab after the per-lab fixes)
V3_CATEGORY = {"glucose_serum": "glucose", "glucose_fingerstick": "glucose"}


def unit_21(category: str) -> str:
    spec = LABS.get(category)
    return "" if spec is None or spec.unit_21 is None else spec.unit_21
