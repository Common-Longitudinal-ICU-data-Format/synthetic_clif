"""Respiratory support table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import HospitalCourse, course_for_row
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class RespiratoryGenerator(BaseGenerator):
    """Generate synthetic respiratory support data.

    Creates respiratory_support table with:
    - hospitalization_id (foreign key)
    - recorded_dttm (hourly in ICU, less frequent elsewhere)
    - device_category (IMV, NIPPV, CPAP, High Flow NC, etc.)
    - mode_category (Volume Control, Pressure Support, etc.)
    - Ventilator settings (fio2, peep, tidal_volume, etc.)
    - tracheostomy flag

    Features:
    - Device escalation and weaning follow the shared course
      (synthetic_clif.physiology), with FiO2 titrated to oxygenation
    - Mode, tidal volume (6-8 mL/kg predicted body weight), rate and PEEP
      persist through an IMV episode; PEEP follows the ARDSNet table
    - Observed values follow respiratory mechanics: plateau = PEEP + VT/Crs,
      peak = plateau + resistive pressure, minute ventilation = VT x RR
    """

    # CLIF 2.1.0 mode_category values
    # "Assist Control-Volume Control", "Pressure Control",
    # "Pressure-Regulated Volume Control", "SIMV",
    # "Pressure Support/CPAP", "Volume Support", "Blow by", "Other"

    VENT_BRANDS = ["Puritan Bennett 840", "Servo-i", "Drager V500", "Hamilton G5", "Avea"]

    # Device-specific settings ranges
    # Consortium aggregate mode weights for IMV:
    # AC-VC 59.7%, PRVC 16.8%, SIMV 9.5%, PS/CPAP 6.3%, PC 2.1%, Other 3.1%
    IMV_MODE_WEIGHTS = [0.597, 0.021, 0.168, 0.095, 0.063]

    DEVICE_SETTINGS = {
        "IMV": {
            "modes": [
                "Assist Control-Volume Control",
                "Pressure Control",
                "Pressure-Regulated Volume Control",
                "SIMV",
                "Pressure Support/CPAP",
            ],
            # Consortium: FiO2 median 0.4 [0.3, 0.6] — use beta-like via
            # truncated range; actual sampling uses _sample_fio2() below
            "fio2_range": (0.3, 1.0),
            # Consortium: PEEP median 5 [5, 8]
            "peep_range": (5, 20),
            # Consortium: TV median 450 [400, 500]
            "tidal_volume_range": (350, 550),
            # Consortium: RR median 16 [14, 20]
            "resp_rate_range": (12, 24),
            "pressure_support_range": (5, 20),
            "pressure_control_range": (15, 35),
        },
        "NIPPV": {
            "modes": ["Other", "Pressure Support/CPAP"],
            "fio2_range": (0.3, 0.6),
            "peep_range": (5, 10),
            "pressure_support_range": (8, 20),
        },
        "CPAP": {
            "modes": ["Pressure Support/CPAP"],
            "fio2_range": (0.3, 0.5),
            "peep_range": (5, 10),
        },
        "High Flow NC": {
            "modes": [None],
            "fio2_range": (0.3, 0.8),
            "flow_rate_range": (20, 60),
        },
        "Face Mask": {
            "modes": [None],
            "fio2_range": (0.28, 0.5),
            "lpm_range": (6, 15),
        },
        "Nasal Cannula": {
            "modes": [None],
            "fio2_range": (0.24, 0.44),
            "lpm_range": (1, 6),
        },
        "Trach Collar": {
            "modes": [None],
            "fio2_range": (0.28, 0.5),
        },
        "Room Air": {
            "modes": [None],
            "fio2_range": (0.21, 0.21),
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        vitals_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate respiratory support data for hospitalizations.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            vitals_df: Optional vitals table for SpO2 correlation

        Returns:
            DataFrame with respiratory_support table columns
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

            hosp_resp = self._generate_hospitalization_respiratory(
                hosp_id, admit_time, discharge_time, course_for_row(hosp)
            )
            records.extend(hosp_resp)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

        return df

    def _generate_hospitalization_respiratory(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        course: Optional[HospitalCourse] = None,
    ) -> list[dict]:
        """Chart respiratory support for one hospitalization from its course.

        Devices, modes and settings come from the shared course, so they
        persist between charting times and change only when clinicians would
        change them. Charting is hourly on NIPPV/IMV, about every 4 hours
        otherwise, and always at a device change.
        """
        if course is None:
            course = course_for_row({
                "hospitalization_id": hospitalization_id,
                "admission_dttm": admit_time,
                "discharge_dttm": discharge_time,
            })
        records = []
        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=1.0,
            cv=0.25,
            rng=self.rng,
        )
        device_ids = {}
        last_key = None
        for ts in timestamps:
            h = course.hour_index(ts)
            key = (course.device[h], course.mode[h], int(course.trach[h]))
            charted = key != last_key or course.level[h] >= 3 or self.rng.random() < 0.25
            if not charted:
                continue
            last_key = key
            records.append(self._chart(hospitalization_id, ts, course, h, device_ids))
        return records

    def _chart(self, hospitalization_id, ts, course: HospitalCourse, h: int, device_ids: dict) -> dict:
        """One respiratory_support row observed from the course at hour h."""
        device = course.device[h]
        mode = course.mode[h]
        if device not in device_ids:
            device_ids[device] = f"RESP-{self.rng.integers(1000, 9999)}"

        def val(arr, nd=0):
            v = arr[h]
            return None if np.isnan(v) else round(float(v), nd) if nd else float(round(v))

        record = {
            "hospitalization_id": hospitalization_id,
            "recorded_dttm": ts,
            "device_id": device_ids[device],
            "device_name": device.lower(),
            "device_category": device,
            "vent_brand_name": None,
            "mode_name": mode.lower() if mode else None,
            "mode_category": mode,
            "tracheostomy": int(course.trach[h]),
            "fio2_set": round(float(course.fio2[h]), 2),
            "lpm_set": val(course.lpm),
            "tidal_volume_set": val(course.vt_set),
            "resp_rate_set": val(course.rr_set),
            "pressure_control_set": val(course.pc),
            "pressure_support_set": val(course.ps),
            "flow_rate_set": val(course.flow),
            "peak_inspiratory_pressure_set": None,
            "inspiratory_time_set": val(course.ti_set, 2),
            "peep_set": val(course.peep),
            "tidal_volume_obs": None,
            "resp_rate_obs": None,
            "plateau_pressure_obs": None,
            "peak_inspiratory_pressure_obs": None,
            "peep_obs": None,
            "minute_vent_obs": None,
            "mean_airway_pressure_obs": None,
        }
        if device == "Room Air":
            record["fio2_set"] = 0.21
        if device != "IMV":
            return record

        record["vent_brand_name"] = course.vent_brand
        peep = course.peep[h]
        crs = max(course.crs[h] * (1 + self.rng.normal(0, 0.05)), 10.0)
        rr_obs = max(round(course.at("rr", ts) + self.rng.normal(0, 0.8)), 4)
        if record["resp_rate_set"] is not None:
            rr_obs = max(rr_obs, record["resp_rate_set"])

        if mode == "Pressure Control":
            dp = course.pc[h]
            vt = dp * crs + self.rng.normal(0, 20)
            plateau = peep + dp
            peak = peep + dp + abs(self.rng.normal(1, 0.5))
            record["peak_inspiratory_pressure_set"] = float(peep + dp)
        elif mode == "Pressure Support/CPAP":
            dp = course.ps[h]
            # Patient effort adds to the support pressure
            vt = dp * crs + self.rng.normal(120, 60)
            plateau = None
            peak = peep + dp + abs(self.rng.normal(1, 0.5))
        else:
            vt = course.vt_set[h] + self.rng.normal(0, 15)
            plateau = peep + vt / crs
            # Resistive pressure: constant flow (VC) vs decelerating flow (PRVC)
            flow_lps = 0.4 if mode == "Pressure-Regulated Volume Control" else self.rng.uniform(0.8, 1.0)
            peak = plateau + course.raw * flow_lps + self.rng.normal(0, 1)
        vt = max(vt, 100.0)
        record["tidal_volume_obs"] = float(round(vt))
        record["resp_rate_obs"] = float(rr_obs)
        record["peak_inspiratory_pressure_obs"] = float(round(peak))
        if plateau is not None and self.rng.random() < 0.3:  # inspiratory-hold manoeuvre
            record["plateau_pressure_obs"] = float(round(min(plateau, peak)))
        record["peep_obs"] = float(round(peep + abs(self.rng.normal(0, 0.5))))
        record["minute_vent_obs"] = round(vt * rr_obs / 1000 + self.rng.normal(0, 0.2), 1)
        record["mean_airway_pressure_obs"] = float(round(peep + 0.4 * (peak - peep)))
        return record
