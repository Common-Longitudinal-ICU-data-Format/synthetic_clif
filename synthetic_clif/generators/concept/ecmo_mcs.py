"""ECMO/MCS (Mechanical Circulatory Support) generator.

Support comes from the hospital course: VV ECMO for refractory hypoxaemia,
VA ECMO for refractory shock, and an Impella or balloon pump for cardiogenic
shock on inotropes. The configuration stays fixed for the run; flow, sweep
and pump speed drift slowly and are weaned in the final day before
decannulation.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row

MCS_GROUP = {
    "VV_ECMO": "ECMO",
    "VA_ECMO": "ECMO",
    "IABP": "IABP",
    "Impella_CP": "temporary_LVAD",
    "Impella_5.5": "temporary_LVAD",
    "CentriMag_LV": "temporary_LVAD",
}


class ECMOMCSGenerator(BaseGenerator):
    """Generate synthetic ECMO/MCS data from the shared course."""

    COLUMNS = ["hospitalization_id", "recorded_dttm", "device_name", "device_category", "mcs_group",
               "sweep_set", "flow", "fdO2_set", "ecmo_configuration_category",
               "control_parameter_name", "control_parameter_category", "control_parameter_value"]

    def generate(self, hospitalizations_df: pd.DataFrame, ecmo_rate: float = 0.02) -> pd.DataFrame:
        """``ecmo_rate`` is kept for compatibility; the course decides who needs support."""
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            course = course_for_row(hosp)
            if course is None or not course.mcs:
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            for run in course.mcs:
                records.extend(self._run(hosp["hospitalization_id"], admit, discharge, course, run))
        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
        return df

    def _run(self, hosp_id, admit, discharge, course, run) -> list[dict]:
        device = run["device"]
        group = MCS_GROUP.get(device, "ECMO")
        s, e = int(run["start"]), int(run["end"])
        n = max(e - s, 1)
        rng = self.rng
        # time-to-wean: the last 24 h of a run that ends before the stay does
        weaning = e < course.n_hours - 2
        wean = np.zeros(n)
        if weaning:
            k = min(24, n)
            wean[n - k:] = np.linspace(0, 1, k)
        drift = np.cumsum(rng.normal(0, 0.05, n))
        drift -= drift.mean()

        if device == "VV_ECMO":
            flow0 = float(np.clip(0.065 * course.weight_kg, 3.5, 6.0))
            flow = flow0 * (1 + 0.04 * drift)
            sweep = np.clip(rng.uniform(4, 7) * (1 + 0.1 * drift) * (1 - wean), 0, 10)
            fdo2 = np.where(wean > 0.8, 0.21, 1.0)
        elif device == "VA_ECMO":
            flow0 = float(np.clip(0.06 * course.weight_kg, 3.5, 5.5))
            flow = flow0 * (1 + 0.04 * drift) * (1 - 0.55 * wean)
            sweep = np.clip(rng.uniform(2, 4) * (1 + 0.1 * drift), 0.5, 8)
            fdo2 = np.full(n, float(rng.choice([0.6, 0.8, 1.0])))
        elif device.startswith("Impella"):
            p0 = float(rng.integers(6, 9))
            plevel = np.clip(np.round(p0 - (p0 - 2) * wean), 2, 9)
            max_flow = 3.7 if device == "Impella_CP" else 5.5
            flow = np.clip(0.45 + plevel * max_flow / 9 + rng.normal(0, 0.1, n), 0.5, max_flow)
        elif device == "CentriMag_LV":
            flow = np.clip(rng.uniform(3.5, 5) * (1 + 0.04 * drift) * (1 - 0.5 * wean), 1.5, 6)

        rows = []
        for j in range(n):
            i = s + j
            ts = admit + timedelta(hours=i + float(rng.uniform(0, 0.25)))
            if ts >= discharge:
                break
            rec = {
                "hospitalization_id": hosp_id,
                "recorded_dttm": ts,
                "device_name": device.lower(),
                "device_category": device,
                "mcs_group": group,
                "sweep_set": None,
                "flow": None,
                "fdO2_set": None,
                "ecmo_configuration_category": None,
                "control_parameter_name": None,
                "control_parameter_category": None,
                "control_parameter_value": None,
            }
            if group == "ECMO":
                f = float(flow[j] + rng.normal(0, 0.05))
                rec["flow"] = round(f, 2)
                rec["sweep_set"] = round(float(sweep[j]) * 2) / 2
                rec["fdO2_set"] = round(float(fdo2[j]), 2)
                rec["ecmo_configuration_category"] = device[:2]
                rec["control_parameter_name"] = "RPM"
                rec["control_parameter_category"] = "rpm"
                rec["control_parameter_value"] = float(round(650 * f + 600 + rng.normal(0, 40), -1))
            elif device.startswith("Impella"):
                rec["flow"] = round(float(flow[j]), 1)
                rec["control_parameter_name"] = "P-Level"
                rec["control_parameter_category"] = "p-level"
                rec["control_parameter_value"] = float(plevel[j])
            elif device == "CentriMag_LV":
                f = float(flow[j])
                rec["flow"] = round(f, 1)
                rec["control_parameter_name"] = "Speed"
                rec["control_parameter_category"] = "speed"
                rec["control_parameter_value"] = float(round(700 * f + 800, -1))
            elif device == "IABP":
                ratio = 1.0 if wean[j] < 0.4 else 2.0 if wean[j] < 0.8 else 3.0
                rec["control_parameter_name"] = "Ratio"
                rec["control_parameter_category"] = "ratio"
                rec["control_parameter_value"] = ratio
            rows.append(rec)
        return rows
