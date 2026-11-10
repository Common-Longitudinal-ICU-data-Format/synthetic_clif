"""Medication orders generator.

One order per ``med_order_id`` in the administration tables, described by
what was actually given: the starting rate or usual dose, the route, the
frequency seen between doses, PRN status for as-needed drugs, and an end
time and status that follow the last administration and the discharge.
"""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator

# As-needed drugs and the shortest interval the order allows
PRN_MEDS = {"hydromorphone": "Q3H PRN", "morphine": "Q4H PRN", "oxycodone": "Q4H PRN",
            "ondansetron": "Q6H PRN", "diphenhydramine": "Q6H PRN", "lorazepam": "Q4H PRN",
            "acetaminophen": "Q6H PRN"}
ONCE_MEDS = {"ketamine", "etomidate", "rocuronium", "succinylcholine"}
ANTIBIOTICS = {"vancomycin", "piperacillin_tazobactam", "cefepime", "meropenem", "ceftriaxone",
               "metronidazole", "cefazolin", "ampicillin_sulbactam", "azithromycin", "levofloxacin"}

# Allowed scheduled frequencies, in hours
_FREQ = [(4, "Q4H"), (6, "Q6H"), (8, "Q8H"), (12, "Q12H"), (24, "Q24H"), (36, "Q36H"), (48, "Q48H")]


def _frequency(hours: float) -> tuple[float, str]:
    h, label = min(_FREQ, key=lambda f: abs(np.log(f[0] / hours)))
    return float(h), label


class MedicationOrdersGenerator(BaseGenerator):
    """Generate medication orders from the administration tables."""

    COLUMNS = ["hospitalization_id", "med_order_id", "order_start_dttm", "order_end_dttm", "ordered_dttm",
               "med_category", "med_name", "med_group", "med_dose", "med_dose_unit",
               "med_order_status_name", "med_order_status_category", "med_route_name",
               "med_frequency", "prn"]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        med_continuous_df: Optional[pd.DataFrame] = None,
        med_intermittent_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        discharge = {}
        if hospitalizations_df is not None and len(hospitalizations_df):
            discharge = dict(zip(hospitalizations_df["hospitalization_id"],
                                 pd.to_datetime(hospitalizations_df["discharge_dttm"], utc=True)))
        records = []
        seen = set()

        if med_continuous_df is not None and len(med_continuous_df) > 0:
            mc = med_continuous_df.sort_values("admin_dttm")
            for order_id, g in mc.groupby("med_order_id", sort=False):
                seen.add(order_id)
                records.append(self._continuous(order_id, g, discharge))

        if med_intermittent_df is not None and len(med_intermittent_df) > 0:
            mi = med_intermittent_df.sort_values("admin_dttm")
            for order_id, g in mi.groupby("med_order_id", sort=False):
                if order_id in seen:
                    continue
                seen.add(order_id)
                records.append(self._intermittent(order_id, g, discharge))

        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            for col in ("order_start_dttm", "order_end_dttm", "ordered_dttm"):
                df[col] = pd.to_datetime(df[col], utc=True)
        return df

    # ------------------------------------------------------------------
    def _base(self, order_id, g, start):
        first = g.iloc[0]
        return {
            "hospitalization_id": first["hospitalization_id"],
            "med_order_id": order_id,
            "order_start_dttm": start,
            "ordered_dttm": start - timedelta(minutes=float(self.rng.uniform(5, 40))),
            "med_category": first["med_category"],
            "med_name": first.get("med_name"),
            "med_group": first.get("med_group"),
            "med_dose_unit": first.get("med_dose_unit"),
            "med_route_name": first.get("med_route_name") or "IV",
        }

    @staticmethod
    def _status(end, dc, finished: bool) -> str:
        if dc is None or pd.isna(dc):
            return "Active"
        if end >= dc - timedelta(minutes=30):
            return "Discontinued"  # stopped at discharge
        return "Completed" if finished else "Discontinued"

    def _continuous(self, order_id, g, discharge) -> dict:
        start = g["admin_dttm"].iloc[0]
        given = g[g["mar_action_category"] != "stop"]
        dose = given["med_dose"].dropna()
        end = g["admin_dttm"].iloc[-1]
        dc = discharge.get(g["hospitalization_id"].iloc[0])
        if dc is not None and not pd.isna(dc):
            end = max(min(end, dc), start)
        status = self._status(end, dc, finished=False)
        rec = self._base(order_id, g, start)
        rec.update({
            "order_end_dttm": end,
            "med_dose": float(dose.iloc[0]) if len(dose) else None,
            "med_order_status_name": status,
            "med_order_status_category": status.lower(),
            "med_frequency": "Continuous",
            "prn": False,
        })
        return rec

    def _intermittent(self, order_id, g, discharge) -> dict:
        cat = g["med_category"].iloc[0]
        times = pd.to_datetime(g["admin_dttm"], utc=True)
        start, last = times.iloc[0], times.iloc[-1]
        prn = cat in PRN_MEDS or (cat == "acetaminophen" and len(g) > 1
                                  and times.diff().dt.total_seconds().std() / 3600 > 3)
        if len(g) == 1 and (cat in ONCE_MEDS or cat not in ANTIBIOTICS):
            freq_h, freq = 0.0, "Once"
        else:
            gaps = times.diff().dt.total_seconds().dropna() / 3600
            gap = float(np.median(gaps)) if len(gaps) else 24.0
            freq_h, freq = _frequency(max(gap, 1.0))
            if prn:
                freq = PRN_MEDS[cat]
                freq_h = float(freq[1:freq.index("H")])
        end = last + timedelta(hours=freq_h if freq_h else 0.1)
        dc = discharge.get(g["hospitalization_id"].iloc[0])
        if dc is not None and not pd.isna(dc):
            end = max(min(end, dc), last)
        status = self._status(end, dc, finished=cat in ANTIBIOTICS or freq == "Once")
        dose = g["med_dose"].dropna()
        rec = self._base(order_id, g, start)
        rec.update({
            "order_end_dttm": end,
            "med_dose": float(dose.mode().iloc[0]) if len(dose) else None,
            "med_order_status_name": status,
            "med_order_status_category": status.lower(),
            "med_frequency": freq,
            "prn": bool(prn),
        })
        return rec
