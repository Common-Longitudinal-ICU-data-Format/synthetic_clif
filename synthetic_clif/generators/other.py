"""Other beta table generators: code_status, position, crrt_therapy."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import course_for_row
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class CodeStatusGenerator(BaseGenerator):
    """Generate synthetic code status data per CLIF 2.1.0.

    Creates code_status table with:
    - patient_id (not hospitalization_id per schema)
    - start_dttm (not recorded_dttm)
    - code_status_category (Full, DNR, DNR/DNI, etc.)
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate code status changes."""
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            patient_id = hosp.get("patient_id", hosp["hospitalization_id"])
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]
            discharge_category = hosp.get("discharge_category", "")

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            course = course_for_row(hosp)
            if course is None:
                continue
            hosp_codes = self._course_codes(patient_id, admit_time, discharge_time, course)
            records.extend(hosp_codes)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["start_dttm"] = pd.to_datetime(df["start_dttm"], utc=True)

        return df

    def _course_codes(self, patient_id, admit_time, discharge_time, course) -> list[dict]:
        """Code status from the course: older patients arrive DNR more often,
        and most deaths follow a change to DNR, then comfort care."""
        events = list(course.code_status)
        if course.comfort_h is not None:
            t = max(course.comfort_h - self.rng.uniform(0, 1), events[-1][0] + 0.25)
            events.append((t, "Other"))  # comfort measures only
        rows = []
        for h, cat in events:
            ts = min(admit_time + timedelta(hours=float(h)), discharge_time)
            rows.append({
                "patient_id": patient_id,
                "start_dttm": ts,
                "code_status_category": cat,
                "code_status_name": self.name_from_category(cat),
            })
        return rows


class PositionGenerator(BaseGenerator):
    """Generate synthetic patient position data per CLIF 2.1.0.

    Creates position table with position_category: prone, not_prone
    """

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate patient position data."""
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
            records.extend(self._course_positions(hosp_id, admit_time, discharge_time, course))

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _course_positions(self, hosp_id, admit_time, discharge_time, course) -> list[dict]:
        """Position charted every ~2 h, plus each turn into and out of prone."""
        times = list(generate_irregular_timestamps(admit_time, discharge_time, mean_interval_hours=2,
                                                   cv=0.4, rng=self.rng))
        prone = course.prone
        for i in range(1, course.n_hours):
            if prone[i] != prone[i - 1]:
                ts = admit_time + timedelta(hours=i + self.rng.uniform(0, 0.25))
                if ts < discharge_time:
                    times.append(ts)
        rows = []
        for ts in sorted(times):
            cat = "prone" if prone[course.hour_index(ts)] else "not_prone"
            rows.append({
                "hospitalization_id": hosp_id,
                "recorded_dttm": ts,
                "position_category": cat,
                "position_name": self.name_from_category(cat),
            })
        return rows


class CRRTTherapyGenerator(BaseGenerator):
    """Generate synthetic CRRT data per CLIF 2.1.0 schema."""

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        crrt_rate: float = 0.08,
    ) -> pd.DataFrame:
        """Generate CRRT therapy data."""
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
            if course is None or not course.crrt.any():
                continue
            hosp_crrt = self._course_crrt(hosp_id, admit_time, discharge_time, course)
            records.extend(hosp_crrt)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _course_crrt(self, hosp_id, admit_time, discharge_time, course) -> list[dict]:
        """Hourly CRRT charting while the course is on CRRT.

        One machine per stay, a weight-based effluent dose split by mode, and
        net fluid removal held while the patient needs vasopressors.
        """
        mode = course.crrt_mode
        machine = str(self.rng.choice(["Prismaflex", "PrisMax", "NxStage", "Aquarius"]))
        device_id = f"CRRT-{self.rng.integers(1000, 9999)}"
        qb = float(self.rng.choice([150, 180, 200, 220, 250]))
        effluent = round(25 * course.weight_kg / 100) * 100   # mL/h, ~25 mL/kg/h
        uf_goal = float(self.rng.choice([50, 75, 100, 150]))
        rows = []
        for i in np.nonzero(course.crrt)[0]:
            if self.rng.random() < 0.08:   # filter clotted or paused
                continue
            ts = admit_time + timedelta(hours=float(i) + self.rng.uniform(0, 0.3))
            if ts >= discharge_time:
                break
            if self.rng.random() < 0.03:
                qb = float(np.clip(qb + self.rng.choice([-20, 20]), 150, 300))
            uf = 0.0 if course.ne_eq[i] > 0.2 else uf_goal
            rec = {
                "hospitalization_id": hosp_id,
                "recorded_dttm": ts,
                "device_id": device_id,
                "crrt_mode_category": mode,
                "crrt_mode_name": self.name_from_category(mode),
                "dialysis_machine_name": machine,
                "blood_flow_rate": qb,
                "pre_filter_replacement_fluid_rate": None,
                "post_filter_replacement_fluid_rate": None,
                "dialysate_flow_rate": None,
                "ultrafiltration_out": uf,
            }
            if mode == "cvvhd":
                rec["dialysate_flow_rate"] = float(effluent)
            elif mode == "cvvh":
                rec["pre_filter_replacement_fluid_rate"] = float(round(effluent * 0.6, -1))
                rec["post_filter_replacement_fluid_rate"] = float(round(effluent * 0.4, -1))
            else:
                rec["dialysate_flow_rate"] = float(round(effluent * 0.5, -1))
                rec["pre_filter_replacement_fluid_rate"] = float(round(effluent * 0.3, -1))
                rec["post_filter_replacement_fluid_rate"] = float(round(effluent * 0.2, -1))
            rows.append(rec)
        return rows
