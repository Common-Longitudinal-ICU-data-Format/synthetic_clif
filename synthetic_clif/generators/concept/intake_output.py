"""Intake/Output generator.

Everything comes from the hospital course, so fluid balance tells the same
story as the rest of the record:

- crystalloid boluses when shock starts (30 mL/kg for sepsis)
- maintenance fluid while the patient can't eat, stopped on CRRT
- tube feeds that start a day or two into ventilation and ramp to goal
- meals for patients who are eating
- blood products, with the same volumes as the transfusion table
- urine output that falls with kidney injury and shock, charted hourly while
  critically ill and as shift totals otherwise
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row
from synthetic_clif.physiology.events import transfusion_units


class IntakeOutputGenerator(BaseGenerator):
    """Generate synthetic intake/output data from the shared course."""

    COLUMNS = ["hospitalization_id", "intake_dttm", "in_out_flag", "amount", "fluid_name"]

    def generate(self, hospitalizations_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            course = course_for_row(hosp)
            if course is None:
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            records.extend(self._stay(hosp["hospitalization_id"], admit, discharge, course))
        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["intake_dttm"] = pd.to_datetime(df["intake_dttm"], utc=True)
        return df

    def _stay(self, hosp_id, admit, discharge, c) -> list[dict]:
        rows = []
        n = c.n_hours

        def add(h, flag, amount, name):
            ts = admit + timedelta(hours=float(h))
            if ts < discharge and amount > 0:
                rows.append({"hospitalization_id": hosp_id, "intake_dttm": ts, "in_out_flag": flag,
                             "amount": float(round(amount)), "fluid_name": name})

        # Boluses
        for h, fluid, vol in c.fluid_boluses:
            add(h + 0.5, 1, vol, fluid)

        # Maintenance fluid and tube feeds: totals charted every 4 h
        mivf_name = str(self.rng.choice(["Lactated Ringers", "Normal Saline", "D5 1/2NS"], p=[0.4, 0.3, 0.3]))
        if c.ckd or (c.crrt is not None and c.crrt.any()):
            feed_name = "Nepro"
        elif c.diabetic:
            feed_name = "Glucerna"
        else:
            feed_name = str(self.rng.choice(["Jevity", "Osmolite", "Promote"]))
        for b in range(0, n, 4):
            add(b + 4 - 0.1, 1, c.mivf[b:b + 4].sum(), mivf_name)
            add(b + 4 - 0.1, 1, c.tube_feed[b:b + 4].sum(), feed_name)

        # Meals, with a drink
        for h, vol in c.meals:
            add(h + 0.5, 1, vol, "PO Intake")

        # Blood products, charted when each unit finishes
        for u in transfusion_units(c):
            add(u["end_h"], 1, u["volume"], u["component"])

        # Urine: hourly while critically ill, otherwise as 8-hour shift totals
        crit = (c.level >= 2) | (c.ne_eq > 0) | (c.severity > 0.6)
        if c.crrt is not None:
            crit = crit | c.crrt
        h = 0
        while h < n:
            if crit[h]:
                if self.rng.random() > 0.06:  # occasional missed hour
                    add(h + 0.9, 0, c.uop[h], "Urine Output")
                h += 1
            else:
                end = min(h + 8 - (h % 8), n)
                vol = c.uop[h:end].sum()
                if self.rng.random() > 0.1:  # some voids go unmeasured
                    add(end - 0.1, 0, vol * self.rng.uniform(0.85, 1.0), "Urine Output")
                h = end
        return rows
