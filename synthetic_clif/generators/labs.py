"""Labs table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_ordered_timestamps
from synthetic_clif.physiology.course import HospitalCourse, course_for_row


class LabsGenerator(BaseGenerator):
    """Generate synthetic laboratory results.

    Creates labs table with:
    - hospitalization_id (foreign key)
    - lab_order_dttm, lab_collect_dttm, lab_result_dttm (ordered timestamps)
    - lab_category (mCIDE categories)
    - lab_value, lab_value_numeric (with realistic ranges)
    - reference_unit (standard units)
    - lab_order_category (blood_gas, bmp, cbc, coags, lft, misc per CLIF 2.1.0)
    - lab_order_name, lab_name, lab_specimen_name, lab_specimen_category, lab_loinc_code

    Features:
    - Values observed from the shared course (synthetic_clif.physiology):
      each patient has their own baseline, values persist day to day, and
      linked labs move together (creatinine/BUN, lactate/bicarbonate/pH,
      Na - Cl - HCO3 anion gap, Henderson-Hasselbalch pH)
    - Daily routine labs, serial lactates on vasopressors, blood gases on IMV
    - Proper timestamp ordering: order < collect < result
    - ~10% missingness for less common labs
    """

    # LOINC codes by lab category
    LAB_LOINC_CODES = {
        "sodium": "2951-2", "potassium": "2823-3", "chloride": "2075-0",
        "bicarbonate": "1963-8", "bun": "3094-0", "creatinine": "2160-0",
        "glucose_serum": "2345-7", "calcium_total": "17861-6",
        "albumin": "1751-7", "total_protein": "2885-2",
        "ast": "1920-8", "alt": "1742-6", "alkaline_phosphatase": "6768-6",
        "bilirubin_total": "1975-2", "bilirubin_conjugated": "1968-7",
        "hemoglobin": "718-7", "wbc": "6690-2", "platelet_count": "777-3",
        "inr": "6301-6", "pt": "5902-2", "ptt": "3173-2",
        "lactate": "2524-7", "troponin_i": "10839-9", "crp": "1988-5",
        "procalcitonin": "33959-8", "ph_arterial": "2744-1",
        "pco2_arterial": "2019-8", "po2_arterial": "2703-7",
        "ph_venous": "2746-6", "pco2_venous": "2021-4",
        "so2_central_venous": "2716-9",
        "fibrinogen": "3255-7", "d_dimer": "48065-7",
        "magnesium": "19123-9", "phosphate": "2777-1",
    }

    # Panel name to human-readable order name
    PANEL_TO_ORDER_NAME = {
        "basic_metabolic": "Basic Metabolic Panel",
        "comprehensive_metabolic": "Comprehensive Metabolic Panel",
        "cbc": "Complete Blood Count",
        "coagulation": "Coagulation Panel",
        "abg_arterial": "Arterial Blood Gas",
        "abg_venous": "Venous Blood Gas",
        "lactate": "Lactate",
        "liver": "Hepatic Function Panel",
        "cardiac": "Cardiac Panel",
        "inflammatory": "Inflammatory Markers",
    }

    # Lab panels (labs typically ordered together)
    LAB_PANELS = {
        "basic_metabolic": [
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "bun",
            "creatinine",
            "glucose_serum",
        ],
        "comprehensive_metabolic": [
            "sodium",
            "potassium",
            "chloride",
            "bicarbonate",
            "bun",
            "creatinine",
            "glucose_serum",
            "calcium_total",
            "albumin",
            "total_protein",
            "ast",
            "alt",
            "alkaline_phosphatase",
            "bilirubin_total",
        ],
        "cbc": ["hemoglobin", "wbc", "platelet_count"],
        "coagulation": ["inr", "pt", "ptt"],
        "abg_arterial": ["ph_arterial", "pco2_arterial", "po2_arterial", "bicarbonate"],
        "abg_venous": ["ph_venous", "pco2_venous", "so2_central_venous", "bicarbonate"],
        "lactate": ["lactate"],
        "liver": [
            "ast",
            "alt",
            "alkaline_phosphatase",
            "bilirubin_total",
            "bilirubin_conjugated",
            "albumin",
        ],
        "cardiac": ["troponin_i", "procalcitonin"],
        "inflammatory": ["crp", "procalcitonin"],
    }

    # Lab parameters: (mean, std, lower_bound, upper_bound)
    LAB_PARAMS = {
        "sodium": (140, 3, 120, 160),
        "potassium": (4.0, 0.5, 2.5, 7.0),
        "chloride": (102, 3, 90, 115),
        "bicarbonate": (24, 3, 5, 45),
        "bun": (15, 8, 5, 100),
        "creatinine": (1.0, 0.5, 0.3, 10.0),
        "glucose_serum": (110, 40, 40, 500),
        "calcium_total": (9.0, 0.6, 6.0, 14.0),
        "magnesium": (2.0, 0.3, 1.0, 4.0),
        "phosphate": (3.5, 0.8, 1.5, 8.0),
        "albumin": (3.5, 0.6, 1.5, 5.5),
        "total_protein": (7.0, 0.8, 4.0, 10.0),
        "ast": (30, 20, 10, 500),
        "alt": (30, 20, 10, 500),
        "alkaline_phosphatase": (80, 30, 30, 500),
        "bilirubin_total": (0.8, 0.5, 0.1, 20.0),
        "bilirubin_conjugated": (0.2, 0.2, 0, 10.0),
        "lactate": (1.5, 1.0, 0.3, 20.0),
        "hemoglobin": (12, 2, 5, 18),
        "wbc": (8, 4, 0.5, 50),
        "platelet_count": (220, 80, 10, 800),
        "inr": (1.1, 0.3, 0.8, 10.0),
        "pt": (12, 2, 9, 50),
        "ptt": (30, 5, 20, 120),
        "fibrinogen": (300, 80, 100, 800),
        "d_dimer": (300, 200, 0, 10000),
        "troponin_i": (0.02, 0.02, 0, 10),
        "crp": (5, 10, 0, 300),
        "procalcitonin": (0.3, 0.5, 0, 50),
        # Arterial blood gas values
        "ph_arterial": (7.40, 0.05, 6.8, 7.7),
        "pco2_arterial": (40, 5, 15, 120),
        "po2_arterial": (85, 15, 40, 500),
        # Venous blood gas values (slightly different physiology)
        "ph_venous": (7.36, 0.05, 6.8, 7.7),
        "pco2_venous": (46, 5, 20, 130),
        "so2_central_venous": (70, 8, 40, 90),
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
        reference_units = self.mcide.get_lab_reference_units()

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            hosp_labs = self._generate_hospitalization_labs(
                hosp_id, admit_time, discharge_time, reference_units, course_for_row(hosp)
            )
            records.extend(hosp_labs)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["lab_order_dttm"] = pd.to_datetime(df["lab_order_dttm"], utc=True)
            df["lab_collect_dttm"] = pd.to_datetime(df["lab_collect_dttm"], utc=True)
            df["lab_result_dttm"] = pd.to_datetime(df["lab_result_dttm"], utc=True)

            # Add missingness
            df = self.add_missingness(df, "lab_value_numeric", missingness_rate)
            df = self.add_missingness(df, "lab_order_dttm", 0.02)

        return df

    def _generate_hospitalization_labs(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        reference_units: dict[str, str],
        course: Optional[HospitalCourse] = None,
    ) -> list[dict]:
        """Generate all labs for one hospitalization."""
        records = []
        self._course = course
        los_hours = (discharge_time - admit_time).total_seconds() / 3600
        los_days = los_hours / 24

        # Admission labs (comprehensive)
        admission_labs = self._generate_admission_labs(
            hospitalization_id, admit_time, reference_units
        )
        records.extend(admission_labs)

        # Daily routine labs (BMP + CBC)
        current_day = 1
        while current_day < los_days:
            lab_time = admit_time + timedelta(days=current_day, hours=int(self.rng.integers(4, 8)))
            if lab_time >= discharge_time:
                break

            # BMP daily
            records.extend(
                self._generate_panel(
                    hospitalization_id,
                    lab_time,
                    "basic_metabolic",
                    "Routine",
                    reference_units,
                )
            )

            # CBC every 1-2 days
            if current_day % 2 == 0 or self.rng.random() < 0.5:
                records.extend(
                    self._generate_panel(
                        hospitalization_id,
                        lab_time + timedelta(minutes=5),
                        "cbc",
                        "Routine",
                        reference_units,
                    )
                )

            current_day += 1

        # PRN labs (lactate, ABG, coags)
        n_prn = int(los_days * self.rng.uniform(0.5, 2))
        for _ in range(n_prn):
            prn_time = admit_time + timedelta(hours=self.rng.uniform(0, los_hours))
            # 70% arterial, 30% venous blood gas draws (ICU practice)
            abg_panel = self.rng.choice(
                ["abg_arterial", "abg_venous"], p=[0.7, 0.3]
            )
            panel = self.rng.choice(["lactate", abg_panel, "coagulation", "cardiac"])
            lab_type = self.rng.choice(["STAT", "Point of Care"], p=[0.7, 0.3])

            records.extend(
                self._generate_panel(
                    hospitalization_id, prn_time, panel, lab_type, reference_units
                )
            )

        # Course-driven draws: serial lactates on vasopressors, blood gases on IMV
        if course is not None:
            for start, end in _runs(course.ne_eq > 0):
                h = start + self.rng.uniform(0.5, 2)
                while h < min(end, course.n_hours - 2) and h < los_hours:
                    records.extend(self._generate_panel(
                        hospitalization_id, admit_time + timedelta(hours=float(h)),
                        "lactate", "STAT", reference_units))
                    h += self.rng.uniform(4, 8)
            for start, end in course.imv_episodes:
                h = start + self.rng.uniform(0.5, 1.5)   # post-intubation gas
                while h < min(end, course.n_hours - 2) and h < los_hours:
                    records.extend(self._generate_panel(
                        hospitalization_id, admit_time + timedelta(hours=float(h)),
                        "abg_arterial", "STAT", reference_units))
                    h += self.rng.uniform(8, 16)
        self._course = None
        return records

    def _generate_admission_labs(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        reference_units: dict[str, str],
    ) -> list[dict]:
        """Generate comprehensive admission labs."""
        records = []

        # Comprehensive metabolic panel
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time,
                "comprehensive_metabolic",
                "STAT",
                reference_units,
            )
        )

        # CBC
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time + timedelta(minutes=2),
                "cbc",
                "STAT",
                reference_units,
            )
        )

        # Coagulation
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time + timedelta(minutes=4),
                "coagulation",
                "STAT",
                reference_units,
            )
        )

        # ABG (for ICU admits, ~50%); 70% arterial, 30% venous
        if self.rng.random() < 0.5:
            abg_panel = self.rng.choice(
                ["abg_arterial", "abg_venous"], p=[0.7, 0.3]
            )
            records.extend(
                self._generate_panel(
                    hospitalization_id,
                    admit_time + timedelta(minutes=10),
                    abg_panel,
                    "STAT",
                    reference_units,
                )
            )

        # Lactate
        records.extend(
            self._generate_panel(
                hospitalization_id,
                admit_time + timedelta(minutes=6),
                "lactate",
                "STAT",
                reference_units,
            )
        )

        return records

    # Map panel names to CLIF 2.1.0 lab_order_category values
    PANEL_TO_ORDER_CATEGORY = {
        "basic_metabolic": "bmp",
        "comprehensive_metabolic": "bmp",
        "cbc": "cbc",
        "coagulation": "coags",
        "abg_arterial": "blood_gas",
        "abg_venous": "blood_gas",
        "lactate": "blood_gas",
        "liver": "lft",
        "cardiac": "misc",
        "inflammatory": "misc",
    }

    def _generate_panel(
        self,
        hospitalization_id: str,
        order_time: datetime,
        panel_name: str,
        lab_type: str,
        reference_units: dict[str, str],
    ) -> list[dict]:
        """Generate a lab panel."""
        records = []
        labs = self.LAB_PANELS.get(panel_name, [panel_name])
        order_category = self.PANEL_TO_ORDER_CATEGORY.get(panel_name, "misc")

        # Generate ordered timestamps: order -> collect -> result
        collect_delay = int(self.rng.integers(5, 30))  # minutes
        result_delay = int(self.rng.integers(30, 180))  # minutes

        collect_time = order_time + timedelta(minutes=collect_delay)
        result_time = collect_time + timedelta(minutes=result_delay)

        course = getattr(self, "_course", None)
        observed = course.labs_at(collect_time, labs, self.rng) if course is not None else {}

        for lab_cat in labs:
            if lab_cat not in self.LAB_PARAMS:
                continue

            mean, std, lower, upper = self.LAB_PARAMS[lab_cat]
            value = observed.get(lab_cat)
            if value is None:
                value = self.rng.normal(mean, std)
            value = float(np.clip(value, lower, upper))

            # Format value string
            if lab_cat in ["ph", "ph_arterial", "ph_venous"]:
                value_str = f"{value:.2f}"
                value = round(value, 2)
            elif lab_cat in ["troponin_i", "troponin_t", "procalcitonin"]:
                value_str = f"{value:.3f}"
                value = round(value, 3)
            else:
                value_str = f"{value:.1f}"
                value = round(value, 1)

            # Map lab_category to lab_order_category per CLIF 2.1.0 schema
            lab_order_category = self._get_lab_order_category(lab_cat)

            # Derive specimen and order name
            specimen_name, specimen_category = self._get_specimen(lab_cat, panel_name)

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "lab_order_dttm": order_time,
                    "lab_collect_dttm": collect_time,
                    "lab_result_dttm": result_time,
                    "lab_order_category": lab_order_category,
                    "lab_order_name": self.PANEL_TO_ORDER_NAME.get(panel_name, panel_name),
                    "lab_category": lab_cat,
                    "lab_name": lab_cat.lower(),
                    "lab_value": value_str,
                    "lab_value_numeric": value,
                    "reference_unit": reference_units.get(lab_cat, ""),
                    "lab_loinc_code": self.LAB_LOINC_CODES.get(lab_cat, ""),
                    "lab_specimen_name": specimen_name,
                    "lab_specimen_category": specimen_category,
                }
            )

        return records

    def _get_specimen(self, lab_cat: str, panel_name: str) -> tuple[str, str]:
        """Return (specimen_name, specimen_category) based on lab and panel."""
        if panel_name in ("abg_arterial",):
            return "Arterial Blood", "blood"
        elif panel_name in ("abg_venous",):
            return "Venous Blood", "blood"
        elif panel_name == "cbc":
            return "Whole Blood", "blood"
        else:
            name = self.rng.choice(["Serum", "Plasma"])
            return name, "blood"

    def _get_lab_order_category(self, lab_cat: str) -> str:
        """Map lab_category to lab_order_category per CLIF 2.1.0 schema.
        
        Permissible values: blood_gas, bmp, cbc, coags, lft, misc
        """
        blood_gas_labs = {"ph_arterial", "ph_venous", "pco2_arterial", "po2_arterial",
                          "pco2_venous", "so2_arterial", "so2_mixed_venous",
                          "so2_central_venous"}
        bmp_labs = {"sodium", "potassium", "chloride", "bicarbonate", "bun", "creatinine",
                    "glucose_serum", "glucose_fingerstick", "calcium_total",
                    "calcium_ionized", "magnesium", "phosphate"}
        cbc_labs = {"hemoglobin", "wbc", "platelet_count",
                    "basophils_percent", "basophils_absolute", "eosinophils_percent",
                    "eosinophils_absolute", "lymphocytes_percent", "lymphocytes_absolute",
                    "monocytes_percent", "monocytes_absolute", "neutrophils_percent",
                    "neutrophils_absolute"}
        coags_labs = {"inr", "pt", "ptt", "fibrinogen", "d_dimer"}
        lft_labs = {"ast", "alt", "alkaline_phosphatase", "bilirubin_total",
                    "bilirubin_conjugated", "bilirubin_unconjugated",
                    "albumin", "total_protein", "ldh"}

        if lab_cat in blood_gas_labs:
            return "blood_gas"
        elif lab_cat in bmp_labs:
            return "bmp"
        elif lab_cat in cbc_labs:
            return "cbc"
        elif lab_cat in coags_labs:
            return "coags"
        elif lab_cat in lft_labs:
            return "lft"
        else:
            return "misc"


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
