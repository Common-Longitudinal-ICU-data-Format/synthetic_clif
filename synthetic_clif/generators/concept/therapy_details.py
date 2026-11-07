"""Therapy details generator.

Sessions follow the orders and the course. PT, OT and speech start with an
evaluation after their order in the key ICU orders table, then treatment
sessions on days the patient is stable (off the ventilator and pressors).
Respiratory therapy treats patients on oxygen support above a nasal
cannula, and those on the ventilator.
"""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row

ORDER_TO_THERAPY = {
    "PT Evaluation": ("Physical Therapy", 0.85, (20, 45)),
    "OT Evaluation": ("Occupational Therapy", 0.5, (20, 40)),
    "Speech Evaluation": ("Speech Therapy", 0.5, (15, 30)),
}


class TherapyDetailsGenerator(BaseGenerator):
    """Generate therapy sessions from the orders and the shared course."""

    COLUMNS = ["hospitalization_id", "session_start_dttm", "therapy_element_name",
               "therapy_element_category", "therapy_element_value"]

    def generate(self, hospitalizations_df: pd.DataFrame,
                 orders_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        orders = {}
        if orders_df is not None and len(orders_df):
            o = orders_df[orders_df["order_category"].isin(ORDER_TO_THERAPY)]
            for hid, g in o.groupby("hospitalization_id"):
                orders[hid] = list(zip(g["order_dttm"], g["order_category"]))
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            c = course_for_row(hosp)
            if c is None:
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            hid = hosp["hospitalization_id"]
            for ts, therapy, kind, minutes in self._sessions(c, admit, discharge, orders.get(hid, [])):
                records.append({
                    "hospitalization_id": hid,
                    "session_start_dttm": ts,
                    "therapy_element_name": f"{therapy} {kind}",
                    "therapy_element_category": therapy,
                    "therapy_element_value": f"{minutes} minutes",
                })
        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["session_start_dttm"] = pd.to_datetime(df["session_start_dttm"], utc=True)
        return df

    def _daytime(self, t):
        """Next daytime slot (08:00-16:00) at or after t."""
        t = pd.Timestamp(t)
        if t.hour >= 16:
            t = (t + timedelta(days=1)).normalize() + timedelta(hours=8)
        elif t.hour < 8:
            t = t.normalize() + timedelta(hours=8)
        return t + timedelta(minutes=float(self.rng.uniform(0, 90)))

    def _sessions(self, c, admit, discharge, orders):
        rng = self.rng
        out = []
        stable = (c.level < 4) & (c.ne_eq == 0)
        for order_time, order_cat in sorted(orders):
            therapy, per_day, (lo, hi) = ORDER_TO_THERAPY[order_cat]
            t = self._daytime(order_time + timedelta(hours=float(rng.uniform(1, 18))))
            kind = "Evaluation"
            while t < discharge:
                i = min(c.hour_index(t), c.n_hours - 1)
                if kind == "Evaluation" or (stable[i] and rng.random() < per_day):
                    minutes = int(rng.integers(lo, hi))
                    out.append((t, therapy, kind, minutes))
                    kind = "Re-evaluation" if rng.random() < 0.08 else "Treatment"
                t = self._daytime(t.normalize() + timedelta(days=1, hours=8))

        # Respiratory therapy: twice a day on high-flow, NIPPV or the ventilator
        need = c.level >= 2
        for d0 in range(0, c.n_hours, 12):
            if need[d0:d0 + 12].any():
                h = d0 + int(np.argmax(need[d0:d0 + 12])) + rng.uniform(0, 2)
                t = admit + timedelta(hours=float(h))
                if t < discharge:
                    kind = "Evaluation" if not out or all(x[1] != "Respiratory Therapy" for x in out) else "Treatment"
                    out.append((t, "Respiratory Therapy", kind, int(rng.integers(10, 30))))
        return out
