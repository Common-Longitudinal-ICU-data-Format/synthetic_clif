"""Key ICU orders generator.

Consults follow the hospital course:

- PT and OT once the patient is stable enough to mobilise (off the
  ventilator and pressors, awake)
- speech and swallow evaluation after a long ventilation
- nutrition consult when tube feeds start, or for a long stay
- palliative care for patients who die or move to comfort care, and for
  older patients with a DNR order and severe illness
- social work for a discharge anywhere other than home
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row

HOME = {"home", "against medical advice (ama)", ""}


class KeyICUOrdersGenerator(BaseGenerator):
    """Generate PT/OT/consult orders from the shared course."""

    COLUMNS = ["hospitalization_id", "order_dttm", "order_category", "order_name", "order_status_name"]

    def generate(self, hospitalizations_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            c = course_for_row(hosp)
            if c is None:
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            dispo = str(hosp.get("discharge_category") or "").strip().lower()
            for h, cat in self._orders(c, dispo):
                ts = admit + timedelta(hours=float(h))
                if ts >= discharge:
                    continue
                left = (discharge - ts).total_seconds() / 3600
                if left > 24:
                    status = "Completed" if self.rng.random() < 0.92 else "Discontinued"
                else:
                    status = "Active" if self.rng.random() < 0.7 else "Discontinued"
                records.append({
                    "hospitalization_id": hosp["hospitalization_id"],
                    "order_dttm": ts,
                    "order_category": cat,
                    "order_name": cat,
                    "order_status_name": status,
                })
        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)
        return df

    def _orders(self, c, dispo) -> list[tuple[float, str]]:
        rng = self.rng
        n = c.n_hours
        last = c.comfort_h if c.comfort_h is not None else n - 2
        out = []
        critical = bool((c.level >= 3).any() or (c.ne_eq > 0).any())

        # Mobility: first hour after 12 h that the patient is stable and awake
        stable = (c.level < 4) & (c.ne_eq == 0) & (c.rass >= -1)
        idx = np.nonzero(stable[12:last])[0]
        p_mob = 0.85 if (critical or c.age >= 70) else 0.5
        if len(idx) and rng.random() < p_mob:
            h = 12 + int(idx[0]) + rng.uniform(1, 12)
            out.append((h, "PT Evaluation"))
            if rng.random() < 0.8:
                out.append((h + rng.uniform(0, 12), "OT Evaluation"))

        # Swallow evaluation after extubation from a long ventilation
        for s, e in c.imv_episodes:
            if e - s > 48 and e < last and rng.random() < 0.7:
                out.append((e + rng.uniform(2, 24), "Speech Evaluation"))
                break
        else:
            if c.age >= 75 and rng.random() < 0.1:
                out.append((rng.uniform(12, max(13, min(72, last))), "Speech Evaluation"))

        # Nutrition: when tube feeds start, else for a long stay
        fed = np.nonzero(c.tube_feed > 0)[0]
        if len(fed) and rng.random() < 0.95:
            out.append((max(fed[0] - rng.uniform(1, 8), 0), "Nutrition Consult"))
        elif n > 120 and rng.random() < 0.5:
            out.append((rng.uniform(24, 96), "Nutrition Consult"))

        # Palliative care
        dnr = [t for t, s in c.code_status if s.upper().startswith("DNR")]
        if c.expired and c.comfort_h is not None and rng.random() < 0.85:
            out.append((max(c.comfort_h - rng.uniform(4, 36), 1), "Palliative Care Consult"))
        elif c.expired and rng.random() < 0.5:
            out.append((max(n - rng.uniform(12, 72), 1), "Palliative Care Consult"))
        elif dnr and c.age >= 70 and float(c.severity.max()) > 0.8 and rng.random() < 0.5:
            out.append((dnr[0] + rng.uniform(2, 48), "Palliative Care Consult"))
        elif dispo == "hospice":
            out.append((max(n - rng.uniform(24, 72), 1), "Palliative Care Consult"))

        # Discharge planning
        if not c.expired:
            p_sw = 0.3 if dispo in HOME else 0.9
            if rng.random() < p_sw:
                out.append((max(n - rng.uniform(24, 96), rng.uniform(6, 24)), "Social Work Consult"))
        elif rng.random() < 0.4:
            out.append((rng.uniform(6, max(7, n - 6)), "Social Work Consult"))
        return sorted(out)
