"""CLIF 3.0 ``intermittent_dialysis`` table.

Sessions come from the course (``physiology/events.py::_ihd``): patients with
severe kidney injury who never go on CRRT are dialysed for 3-4 hours about
every other day once they are off vasopressors. Each session is charted every
15-30 minutes with the blood and dialysate flow and the cumulative net
ultrafiltration, which ends at the session's goal. The same sessions clear
creatinine and potassium in the labs table, appear in ``patient_procedures``
(ICD-10-PCS 5A1D70Z / CPT 90935), need a dialysis catheter in ``line``, and
put their final ultrafiltrate in ``output``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import at, iter_courses, stream


class IntermittentDialysisGenerator(BaseGenerator):
    COLUMNS = ["hospitalization_id", "device_id", "recorded_dttm", "blood_flow_rate",
               "dialysate_flow_rate", "net_ultrafiltration_out"]

    def generate(self, hospitalizations_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (intermittent_dialysis rows, ultrafiltration output rows)."""
        rows, uf = [], []
        for row, c, admit, discharge in iter_courses(hospitalizations_df):
            if not c.ihd_sessions:
                continue
            hid = row["hospitalization_id"]
            r = stream("ihd_table", c.rng_seed)
            for k, s in enumerate(c.ihd_sessions):
                t, end = s["start"], s["end"]
                times = [t]
                while times[-1] + 0.25 < end:
                    times.append(times[-1] + float(r.choice([0.25, 0.5], p=[0.4, 0.6])))
                times[-1] = min(times[-1], end)
                dur = end - t
                for j, h in enumerate(times):
                    frac = (h - t) / dur if dur > 0 else 1.0
                    # flow is ramped up over the first minutes, then held
                    bfr = s["blood_flow"] if j else s["blood_flow"] - float(r.choice([50, 100, 150]))
                    bfr += float(r.choice([0, 0, 0, -25, 25]))
                    uf_now = s["uf_goal"] * frac * (1 + r.normal(0, 0.03)) if j else 0.0
                    rows.append({
                        "hospitalization_id": hid,
                        "device_id": s["device"],
                        "recorded_dttm": at(admit, h),
                        "blood_flow_rate": float(np.clip(bfr, 150, 450)),
                        "dialysate_flow_rate": s["dialysate_flow"] if j else 0.0,
                        "net_ultrafiltration_out": float(round(max(uf_now, 0.0), 0)),
                    })
                uf.append({
                    "hospitalization_id": hid,
                    "recorded_dttm": at(admit, end + r.uniform(0.02, 0.2)),
                    "output_name": "Hemodialysis Net UF",
                    "output_category": "ultrafiltration",
                    "output_value": float(round(rows[-1]["net_ultrafiltration_out"], 0)),
                    "output_unit": "mL",
                })
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        if len(df):
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
        out = pd.DataFrame(uf)
        if len(out):
            out["recorded_dttm"] = pd.to_datetime(out["recorded_dttm"], utc=True)
        return df, out
