"""Invasive hemodynamics generator.

Readings exist only while the course has a pulmonary-artery catheter in
place. They come from the course, so they agree with MAP and shock type:
low cardiac index with a high wedge pressure in cardiogenic shock, high
output with a low SVR in distributive shock, and SvO2 that falls as shock
worsens.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row

MEASURE_GROUP = {
    "CVP": "pressure",
    "PA Systolic": "pressure",
    "PA Diastolic": "pressure",
    "PA Mean": "pressure",
    "PCWP": "pressure",
    "Cardiac Output": "flow",
    "Cardiac Index": "flow",
    "SVR": "resistance",
    "PVR": "resistance",
    "SvO2": "saturation",
}


class InvasiveHemodynamicsGenerator(BaseGenerator):
    """Generate synthetic invasive hemodynamic data from the shared course."""

    COLUMNS = ["hospitalization_id", "recorded_dttm", "measure_name", "measure_category", "measure_value"]

    def generate(self, hospitalizations_df: pd.DataFrame, pa_catheter_rate: float = 0.08) -> pd.DataFrame:
        """``pa_catheter_rate`` is kept for compatibility; the course places the catheters."""
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            course = course_for_row(hosp)
            if course is None or course.pa_catheter is None or not course.pa_catheter.any():
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            records.extend(self._readings(hosp["hospitalization_id"], admit, discharge, course))
        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
        return df

    def _readings(self, hosp_id, admit, discharge, course) -> list[dict]:
        rows = []
        hours = np.nonzero(course.pa_catheter)[0]
        h = float(hours[0]) + self.rng.uniform(0.5, 1.5)
        last = float(hours[-1]) + 1
        while h < last:
            if course.pa_catheter[int(h)]:
                ts = admit + timedelta(hours=h)
                if ts >= discharge:
                    break
                for name, value in course.hemodynamics_at(ts, self.rng).items():
                    rows.append({
                        "hospitalization_id": hosp_id,
                        "recorded_dttm": ts,
                        "measure_name": name,
                        "measure_category": MEASURE_GROUP[name],
                        "measure_value": round(float(value), 1),
                    })
            # q4h with drift; more often while on high-dose pressors
            step = 2.0 if course.ne_eq[int(h)] > 0.25 else 4.0
            h += step + self.rng.uniform(-0.5, 0.75)
        return rows
