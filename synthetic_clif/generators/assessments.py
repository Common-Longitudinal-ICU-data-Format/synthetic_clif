"""Patient assessments generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import course_for_row
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class PatientAssessmentsGenerator(BaseGenerator):
    """Generate synthetic patient assessment data.

    Creates patient_assessments table with:
    - GCS (total and components)
    - RASS (sedation scale)
    - CAM-ICU (delirium screening)
    - Pain scores
    - Other ICU assessments
    """

    # Assessment groups as clifpy's 2.1 schema maps the categories
    ASSESSMENT_GROUPS = {
        "gcs_total": "Neurological",
        "gcs_eye": "Neurological",
        "gcs_verbal": "Neurological",
        "gcs_motor": "Neurological",
        "RASS": "Sedation/Agitation",
        "cam_total": "Delirium",
        "NRS": "Pain",
        "braden_total": "Nursing Risk",
        "Morse Fall Scale": "Nursing Risk",
    }

    # Assessment parameters
    ASSESSMENT_PARAMS = {
        "gcs_total": {
            "range": (3, 15),
            "default": 15,
            "frequency_hours": 4,
        },
        "gcs_eye": {
            "range": (1, 4),
            "default": 4,
            "frequency_hours": 4,
        },
        "gcs_verbal": {
            "range": (1, 5),
            "default": 5,
            "frequency_hours": 4,
        },
        "gcs_motor": {
            "range": (1, 6),
            "default": 6,
            "frequency_hours": 4,
        },
        "RASS": {
            "range": (-5, 4),
            "default": 0,
            "frequency_hours": 4,
        },
        "cam_total": {
            "range": (0, 1),  # 0=negative, 1=positive
            "default": 0,
            "frequency_hours": 12,
        },
        "NRS": {
            "range": (0, 10),
            "default": 0,
            "frequency_hours": 4,
        },
        "braden_total": {
            "range": (6, 23),
            "default": 18,
            "frequency_hours": 24,
        },
        "Morse Fall Scale": {
            "range": (0, 125),
            "default": 25,
            "frequency_hours": 24,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate patient assessments.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            respiratory_df: Optional respiratory support for ventilation status

        Returns:
            DataFrame with patient_assessments columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            course = course_for_row(hosp)
            records.extend(self._course_assessments(
                hosp_id, admit_time, discharge_time, course,
                str(hosp.get("admission_type_category") or ""),
            ))

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
            # Add text_value column as string type (CLIFPy requires VARCHAR)
            if "text_value" not in df.columns:
                df["text_value"] = pd.array([None] * len(df), dtype="string")

        return df

    def _rec(self, hosp_id, ts, cat, value, text=None) -> dict:
        return {
            "hospitalization_id": hosp_id,
            "recorded_dttm": ts,
            "assessment_category": cat,
            "assessment_name": self.name_from_category(cat),
            "assessment_group": self.ASSESSMENT_GROUPS.get(cat, "Neurological"),
            "numerical_value": None if value is None else float(value),
            "categorical_value": text,
        }

    def _course_assessments(self, hosp_id, admit_time, discharge_time, course, admission_type) -> list[dict]:
        """Scores observed from the course's sedation, delirium and illness.

        RASS and GCS follow sedation depth (which also sets sedative doses),
        CAM-ICU is unassessable when RASS is -4 or lower and positive during
        delirium, pain is scored only in patients who can self-report, and
        Braden and Morse risk rise with immobility, age and delirium.
        """
        rec = []
        n = course.n_hours
        sedated = (course.level == 4) & (course.trach == 0)

        def jitter(v, lo, hi, p=0.12):
            if self.rng.random() < p:
                v += self.rng.choice([-1, 1])
            return int(np.clip(v, lo, hi))

        # Neuro checks: GCS q4h, RASS q2h while sedated on the ventilator
        for ts in generate_irregular_timestamps(admit_time, discharge_time, mean_interval_hours=2,
                                                cv=0.25, rng=self.rng):
            h = course.hour_index(ts)
            rass = jitter(course.rass[h], -5, 4, 0.1)
            if sedated[h] or self.rng.random() < 0.5:
                rec.append(self._rec(hosp_id, ts, "RASS", rass))
            if self.rng.random() < 0.5:
                e = jitter(course.gcs_eye[h], 1, 4, 0.08)
                v = int(course.gcs_verbal[h]) if course.level[h] == 4 else jitter(course.gcs_verbal[h], 1, 5, 0.08)
                m = jitter(course.gcs_motor[h], 1, 6, 0.08)
                for cat, val in (("gcs_total", e + v + m), ("gcs_eye", e), ("gcs_verbal", v), ("gcs_motor", m)):
                    rec.append(self._rec(hosp_id, ts, cat, val))
            # Self-reported pain only when awake and not intubated
            if course.level[h] < 4 and rass >= -1 and self.rng.random() < 0.5:
                pain = (1.5 + 3.0 * (admission_type == "elective") * (h < 72)
                        + 1.5 * min(course.severity[h], 1.0) + self.rng.normal(0, 1.8))
                rec.append(self._rec(hosp_id, ts, "NRS", int(np.clip(round(pain), 0, 10))))

        # CAM-ICU twice a day
        for ts in generate_irregular_timestamps(admit_time, discharge_time, mean_interval_hours=12,
                                                cv=0.3, rng=self.rng):
            h = course.hour_index(ts)
            if course.rass[h] <= -4:
                rec.append(self._rec(hosp_id, ts, "cam_total", None, "Unable to Assess"))
            elif course.delirium[h]:
                rec.append(self._rec(hosp_id, ts, "cam_total", 1, "Positive"))
            else:
                rec.append(self._rec(hosp_id, ts, "cam_total", 0, "Negative"))

        # Daily nursing risk scores
        for ts in generate_irregular_timestamps(admit_time, discharge_time, mean_interval_hours=24,
                                                cv=0.2, rng=self.rng):
            h = course.hour_index(ts)
            immobile = course.level[h] == 4 or course.rass[h] <= -2
            braden = (20 - 6 * immobile - 2 * min(course.severity[h], 1.0) - 1.5 * (course.age >= 75)
                      - 1.5 * course.delirium[h] + self.rng.normal(0, 1))
            rec.append(self._rec(hosp_id, ts, "braden_total", int(np.clip(round(braden), 6, 23))))
            morse = (15 + 20 * (course.age >= 70) + 20 * course.delirium[h] + 20 * bool(course.central_line[h] or course.art_line[h])
                     + 15 * (course.severity[h] > 0.5) + self.rng.normal(0, 8))
            rec.append(self._rec(hosp_id, ts, "Morse Fall Scale", int(np.clip(5 * round(morse / 5), 0, 125))))
        return rec
