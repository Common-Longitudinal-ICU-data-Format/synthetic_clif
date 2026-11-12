"""Labs table generator."""

from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.lab_catalog import LABS
from synthetic_clif.physiology.course import HospitalCourse, course_for_row


class LabsGenerator(BaseGenerator):
    """Generate synthetic laboratory results.

    Creates labs table with:
    - hospitalization_id (foreign key)
    - lab_order_dttm, lab_collect_dttm, lab_result_dttm (ordered timestamps)
    - lab_category: all 52 CLIF 2.1.0 categories, plus the CLIF 3.0.0-only
      categories in ``lab_catalog.CLIF3_ONLY`` (dropped from 2.1.0 output)
    - lab_value, lab_value_numeric, reference_unit (mCIDE units, see
      ``synthetic_clif/config/lab_catalog.py``)
    - lab_order_category, lab_order_name, lab_name, lab_specimen_name,
      lab_specimen_category, lab_loinc_code

    Features:
    - Values observed from the shared course (synthetic_clif.physiology):
      each patient has their own baseline, values persist day to day, and
      linked labs move together (creatinine/BUN, lactate/bicarbonate/pH,
      Na - Cl - HCO3 anion gap, Henderson-Hasselbalch pH, a differential that
      sums to 100% with absolute counts that match the WBC, conjugated +
      unconjugated = total bilirubin, SaO2 from PaO2, ionized from total calcium)
    - Ordering follows the course: daily BMP/CBC with magnesium and phosphate,
      serial lactates on vasopressors, blood gases on IMV, mixed venous gases
      with a PA catheter, fingerstick glucose on insulin, inflammatory markers
      when infection is suspected, serial troponins in cardiogenic shock,
      DIC labs in shock or bleeding, triglycerides on propofol
    - Proper timestamp ordering: order < collect < result
    """

    LAB_LOINC_CODES = {k: v.loinc for k, v in LABS.items()}
    LAB_PARAMS = {k: v.params for k, v in LABS.items()}

    PANEL_TO_ORDER_NAME = {
        "basic_metabolic": "Basic Metabolic Panel",
        "comprehensive_metabolic": "Comprehensive Metabolic Panel",
        "mg_phos": "Magnesium and Phosphorus",
        "cbc": "Complete Blood Count",
        "cbc_diff": "CBC with Differential",
        "coagulation": "Coagulation Panel",
        "dic": "Fibrinogen and D-Dimer",
        "abg_arterial": "Arterial Blood Gas",
        "abg_venous": "Venous Blood Gas",
        "mixed_venous": "Mixed Venous Blood Gas",
        "lactate": "Lactate",
        "liver": "Hepatic Function Panel",
        "cardiac": "Cardiac Biomarkers",
        "inflammatory": "Inflammatory Markers",
        "fingerstick": "POC Glucose",
        "triglycerides": "Triglycerides",
    }

    _CBC = ["hemoglobin", "hematocrit", "wbc", "platelet_count"]
    _DIFF = [f"{c}_{k}" for c in ("neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils")
             for k in ("percent", "absolute")]

    LAB_PANELS = {
        "basic_metabolic": ["sodium", "potassium", "chloride", "bicarbonate", "bun", "creatinine",
                            "glucose_serum"],
        "comprehensive_metabolic": ["sodium", "potassium", "chloride", "bicarbonate", "bun",
                                    "creatinine", "glucose_serum", "calcium_total", "albumin",
                                    "total_protein", "ast", "alt", "alkaline_phosphatase",
                                    "bilirubin_total"],
        "mg_phos": ["magnesium", "phosphate"],
        "cbc": _CBC,
        "cbc_diff": _CBC + _DIFF,
        "coagulation": ["inr", "pt", "ptt"],
        "dic": ["fibrinogen", "d_dimer"],
        # the blood gas analyser reports SaO2 and ionized calcium with the gas
        "abg_arterial": ["ph_arterial", "pco2_arterial", "po2_arterial", "so2_arterial",
                         "bicarbonate", "calcium_ionized"],
        "abg_venous": ["ph_venous", "pco2_venous", "so2_central_venous", "bicarbonate"],
        "mixed_venous": ["so2_mixed_venous", "po2_mixed_venous"],
        "lactate": ["lactate"],
        "liver": ["ast", "alt", "alkaline_phosphatase", "bilirubin_total", "bilirubin_conjugated",
                  "bilirubin_unconjugated", "albumin", "ldh"],
        "cardiac": ["troponin_i"],
        "inflammatory": ["crp", "procalcitonin"],
        "fingerstick": ["glucose_fingerstick"],
        "triglycerides": ["triglycerides"],
    }

    _GASES = ("abg_arterial", "abg_venous", "mixed_venous")

    _SPECIMEN = {
        "abg_arterial": "Arterial Blood",
        "abg_venous": "Venous Blood",
        "mixed_venous": "Mixed Venous Blood",
        "cbc": "Whole Blood",
        "cbc_diff": "Whole Blood",
        "fingerstick": "Capillary Blood",
        "coagulation": "Plasma",
        "dic": "Plasma",
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        missingness_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Generate laboratory results for hospitalizations.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            missingness_rate: Proportion of missing values

        Returns:
            DataFrame with labs table columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]
            if pd.isna(admit_time):
                continue
            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)
            records.extend(self._generate_hospitalization_labs(
                hosp["hospitalization_id"], admit_time, discharge_time, course=course_for_row(hosp)))

        df = pd.DataFrame(records)

        if len(df) > 0:
            for col in ("lab_order_dttm", "lab_collect_dttm", "lab_result_dttm"):
                df[col] = pd.to_datetime(df[col], utc=True)
            df = self.add_missingness(df, "lab_value_numeric", missingness_rate)
            df = self.add_missingness(df, "lab_order_dttm", 0.02)

        return df

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------
    def _generate_hospitalization_labs(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        reference_units: Optional[dict] = None,
        course: Optional[HospitalCourse] = None,
    ) -> list[dict]:
        """Generate all labs for one hospitalization."""
        records: list[dict] = []
        self._course = course
        los_hours = (discharge_time - admit_time).total_seconds() / 3600
        los_days = los_hours / 24
        rng = self.rng

        def draw(hours, panel, lab_type, labs=None):
            if 0 <= hours < los_hours:
                records.extend(self._generate_panel(
                    hospitalization_id, admit_time + timedelta(hours=float(hours)), panel,
                    lab_type, labs=labs))

        ext = course.lab_ext_base if course is not None else {}
        cardiac_labs = [ext.get("troponin", "troponin_i")]
        natriuretic = ext.get("natriuretic", "bnp")

        # Admission labs
        records.extend(self._generate_admission_labs(hospitalization_id, admit_time))
        draw(0.05, "mg_phos", "STAT")
        if rng.random() < 0.15:
            draw(0.08, "liver", "STAT")
        if rng.random() < 0.15:
            draw(0.08, "dic", "STAT")
        if rng.random() < 0.45:
            labs = cardiac_labs + ([natriuretic] if rng.random() < 0.5 else [])
            draw(0.1, "cardiac", "STAT", labs)
        if rng.random() < 0.08:
            draw(0.1, "inflammatory", "STAT", ["crp", "esr"])
        if rng.random() < 0.06:
            draw(0.1, "inflammatory", "STAT", ["ferritin"])

        def sick(day: int) -> bool:
            return course is not None and course.severity[course._i(day * 24)] > 0.35

        # Daily routine labs: BMP (CMP every third day), magnesium and
        # phosphate, CBC (with differential some days), LFTs when sick
        current_day = 1
        while current_day < los_days:
            h = current_day * 24 + float(rng.integers(4, 8))
            if h >= los_hours:
                break
            draw(h, "comprehensive_metabolic" if current_day % 3 == 0 else "basic_metabolic", "Routine")
            if sick(current_day) or rng.random() < 0.5:
                draw(h + 0.05, "mg_phos", "Routine")
            if current_day % 2 == 0 or rng.random() < 0.5:
                draw(h + 0.08, "cbc_diff" if rng.random() < 0.4 else "cbc", "Routine")
            if sick(current_day) and current_day % 2 == 1 and rng.random() < 0.6:
                draw(h + 0.1, "liver", "Routine")
            current_day += 1

        # PRN labs (lactate, blood gas, coags, troponin)
        n_prn = int(los_days * rng.uniform(0.5, 2))
        for _ in range(n_prn):
            prn_h = rng.uniform(0, los_hours)
            abg_panel = rng.choice(["abg_arterial", "abg_venous"], p=[0.7, 0.3])
            panel = str(rng.choice(["lactate", abg_panel, "coagulation", "cardiac"]))
            lab_type = rng.choice(["STAT", "Point of Care"], p=[0.7, 0.3])
            draw(prn_h, panel, lab_type, cardiac_labs if panel == "cardiac" else None)

        if course is not None:
            self._course_driven_labs(course, los_hours, draw, cardiac_labs, natriuretic)
        self._course = None
        return records

    def _course_driven_labs(self, course, los_hours, draw, cardiac_labs, natriuretic) -> None:
        """Draws that follow the course (each one goes through ``draw``)."""
        rng = self.rng
        end_h = min(course.n_hours - 2, los_hours)

        # Serial lactates on vasopressors
        for start, end in _runs(course.ne_eq > 0):
            h = start + rng.uniform(0.5, 2)
            while h < min(end, end_h):
                draw(h, "lactate", "STAT")
                h += rng.uniform(4, 8)

        # Blood gases on IMV
        for start, end in course.imv_episodes:
            h = start + rng.uniform(0.5, 1.5)   # post-intubation gas
            while h < min(end, end_h):
                draw(h, "abg_arterial", "STAT")
                h += rng.uniform(8, 16)

        # Mixed venous gases while a PA catheter is in place
        if course.pa_catheter is not None:
            for start, end in _runs(course.pa_catheter):
                h = start + rng.uniform(0.5, 2)
                while h < min(end, end_h):
                    draw(h, "mixed_venous", "STAT")
                    h += rng.uniform(6, 12)

        # Fingerstick glucose: about hourly on an insulin infusion, every
        # six hours for other diabetic patients
        for start, end in _runs(course.insulin):
            h = start + rng.uniform(0, 1)
            while h < min(end, end_h):
                draw(h, "fingerstick", "Point of Care")
                h += rng.uniform(0.8, 1.4)
        if course.diabetic:
            h = rng.uniform(4, 8)
            while h < end_h:
                if not course.insulin[course._i(h)]:
                    draw(h, "fingerstick", "Point of Care")
                h += rng.uniform(5, 7)

        # Infection suspected: CRP and procalcitonin (sometimes ESR, ferritin,
        # D-dimer), then procalcitonin every one to two days
        if course.infection_recognized_h is not None:
            h0 = float(course.infection_recognized_h) + rng.uniform(0, 1)
            labs = ["crp", "procalcitonin"]
            if rng.random() < 0.3:
                labs.append("esr")
            if rng.random() < 0.3:
                labs.append("ferritin")
            draw(h0, "inflammatory", "STAT", labs)
            if rng.random() < 0.3:
                draw(h0 + 0.1, "dic", "STAT", ["d_dimer"])
            h = h0
            for _ in range(int(rng.integers(0, 4))):
                h += rng.uniform(24, 48)
                draw(h, "inflammatory", "Timed", ["procalcitonin"])

        # Cardiogenic shock: serial troponins and a natriuretic peptide
        shock_on = course.shock > 0.3
        if course.shock_type == "cardiogenic" and shock_on.any():
            h0 = float(np.argmax(shock_on)) + rng.uniform(0, 2)
            draw(h0, "cardiac", "STAT", cardiac_labs + [natriuretic])
            for k in (1, 2):
                draw(h0 + 6 * k + rng.uniform(-0.5, 0.5), "cardiac", "Timed", cardiac_labs)

        # Severe shock or bleeding: fibrinogen, D-dimer and coags every 12-24 h
        bleeding = course.bleed > 1.0 if course.bleed is not None else np.zeros(course.n_hours, bool)
        for start, end in _runs((course.shock > 0.6) | bleeding):
            h = start + rng.uniform(1, 4)
            while h < min(end, end_h):
                draw(h, "dic", "STAT")
                draw(h + 0.02, "coagulation", "STAT")
                h += rng.uniform(12, 24)

        # Triglycerides after two days of propofol, then every three days on it
        prop = course.propofol_hours
        if prop[-1] >= 48:
            h = float(np.argmax(prop >= 48)) + rng.uniform(0, 6)
            while h < end_h:
                draw(h, "triglycerides", "Routine")
                nxt = h + rng.uniform(60, 84)
                if prop[course._i(nxt)] <= prop[course._i(h)]:
                    break  # propofol stopped
                h = nxt

    def _generate_admission_labs(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        reference_units: Optional[dict] = None,
    ) -> list[dict]:
        """Generate comprehensive admission labs."""
        records = []
        records.extend(self._generate_panel(hospitalization_id, admit_time,
                                            "comprehensive_metabolic", "STAT"))
        records.extend(self._generate_panel(hospitalization_id, admit_time + timedelta(minutes=2),
                                            "cbc_diff", "STAT"))
        records.extend(self._generate_panel(hospitalization_id, admit_time + timedelta(minutes=4),
                                            "coagulation", "STAT"))
        # ABG (for ICU admits, ~50%); 70% arterial, 30% venous
        if self.rng.random() < 0.5:
            abg_panel = self.rng.choice(["abg_arterial", "abg_venous"], p=[0.7, 0.3])
            records.extend(self._generate_panel(hospitalization_id,
                                                admit_time + timedelta(minutes=10), abg_panel, "STAT"))
        records.extend(self._generate_panel(hospitalization_id, admit_time + timedelta(minutes=6),
                                            "lactate", "STAT"))
        return records

    # ------------------------------------------------------------------
    # One panel
    # ------------------------------------------------------------------
    def _generate_panel(
        self,
        hospitalization_id: str,
        order_time: datetime,
        panel_name: str,
        lab_type: str,
        reference_units: Optional[dict] = None,
        labs: Optional[Sequence[str]] = None,
    ) -> list[dict]:
        """Generate one panel (``labs`` overrides the panel's usual tests)."""
        records = []
        if labs is None:
            labs = self.LAB_PANELS.get(panel_name, [panel_name])
        labs = [c for c in labs if c in LABS]

        if panel_name == "fingerstick":
            collect_delay, result_delay = 0, int(self.rng.integers(1, 4))
        elif lab_type == "Point of Care" or panel_name in self._GASES:
            collect_delay, result_delay = int(self.rng.integers(2, 15)), int(self.rng.integers(5, 25))
        else:
            collect_delay = int(self.rng.integers(5, 30))
            result_delay = int(self.rng.integers(30, 180))

        collect_time = order_time + timedelta(minutes=collect_delay)
        result_time = collect_time + timedelta(minutes=result_delay)

        course = getattr(self, "_course", None)
        observed = course.labs_at(collect_time, labs, self.rng) if course is not None else {}
        specimen_name = self._SPECIMEN.get(panel_name) or str(self.rng.choice(["Serum", "Plasma"]))
        order_name = self.PANEL_TO_ORDER_NAME.get(panel_name, panel_name)

        for lab_cat in labs:
            spec = LABS[lab_cat]
            mean, std, lower, upper = spec.params
            value = observed.get(lab_cat)
            if value is None:
                value = self.rng.normal(mean, std)
            value = round(float(np.clip(value, lower, upper)), spec.decimals)
            value_str = f"{value:.{spec.decimals}f}"

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "lab_order_dttm": order_time,
                    "lab_collect_dttm": collect_time,
                    "lab_result_dttm": result_time,
                    "lab_order_category": ("blood_gas" if lab_cat == "bicarbonate" and panel_name in self._GASES
                                           else self._get_lab_order_category(lab_cat)),
                    "lab_order_name": order_name,
                    "lab_category": lab_cat,
                    "lab_name": lab_cat.lower(),
                    "lab_value": value_str,
                    "lab_value_numeric": value,
                    "reference_unit": spec.unit_21 or spec.unit_30,
                    "lab_loinc_code": spec.loinc,
                    "lab_specimen_name": specimen_name,
                    "lab_specimen_category": "blood/plasma/serum",
                }
            )

        return records

    def _get_lab_order_category(self, lab_cat: str) -> str:
        """CLIF 2.1.0 lab_order_category from the mCIDE lab list.

        Permissible values: blood_gas, bmp, cbc, coags, lft, misc. CLIF
        3.0.0-only labs carry their 3.0 order category (they never reach
        2.1.0 output).
        """
        spec = LABS.get(lab_cat)
        if spec is None:
            return "misc"
        return spec.order_21 or spec.order_30 or "misc"


def _runs(mask) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive) indices."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs
