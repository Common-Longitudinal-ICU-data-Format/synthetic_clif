"""CLIF 3.0 ``model_registry`` and ``scores`` tables.

The registry lists the EHR predictive models a site runs, one row per
deployed version: the Epic Deterioration Index, the Epic Sepsis Model and
Epic's inpatient fall-risk model. Older versions are kept with
``is_live = 0``. Each score row comes from the version that was deployed on
its date, so ``scores.model_id`` changes when a new version goes live.

Scores follow the course:

- **Deterioration Index** (0-100), about hourly: rises with illness
  severity, oxygen support and vasopressors, so it climbs before ICU
  transfer and intubation and is higher in patients who die.
- **Sepsis Model** (0-100), every 4 hours: driven by the infection and
  severity scores. Version 1 is noisier and more often high in patients
  without infection; version 2 is better calibrated.
- **Fall Risk** (0-100), each nursing shift: rises with age, delirium and
  light sedation, and falls when the patient is bed-bound on a ventilator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import at, iter_courses, stream

REGISTRY = [
    # model_id, model_name, model_category, model_version, release_date, is_live
    ("EDI-001", "IP Deterioration Index", "epic_deterioration_index", "0.0.1", "2017-06-01", 0),
    ("EDI-002", "IP Deterioration Index", "epic_deterioration_index", "0.0.2", "2021-03-15", 1),
    ("ESM-001", "Epic Sepsis Model", "epic_sepsis_model", "1.0", "2017-01-10", 0),
    ("ESM-002", "Sepsis Model v2", "epic_sepsis_model", "2.0", "2022-10-01", 1),
    ("FALL-001", "Epic Fall Risk", "epic_fall_risk", "1.0", "2019-07-01", 1),
]


def model_registry() -> pd.DataFrame:
    df = pd.DataFrame(REGISTRY, columns=["model_id", "model_name", "model_category", "model_version",
                                         "release_date", "is_live"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def _version(category: str, ts: pd.Timestamp):
    """model_id of the version of a model deployed at ts (None before the first)."""
    live = None
    for mid, _, cat, _, rel, _ in REGISTRY:
        if cat == category and ts >= pd.Timestamp(rel, tz="UTC"):
            live = mid
    return live


def _ar(r, n, sd, phi):
    x = np.empty(n)
    x[0] = r.normal(0, sd)
    e = r.normal(0, sd * np.sqrt(1 - phi**2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class ScoresGenerator(BaseGenerator):
    COLUMNS = ["model_id", "hospitalization_id", "recorded_dttm", "score_value"]

    def generate(self, hospitalizations_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row, c, admit, discharge in iter_courses(hospitalizations_df):
            hid = row["hospitalization_id"]
            r = stream("scores", c.rng_seed)
            n = c.n_hours
            last = int(np.floor(c.los_h))
            if last < 1:
                continue
            noise = _ar(r, n, 4.0, 0.8)
            edi = (12 + 45 * np.clip(c.severity, 0, 1.2) + 4 * c.level + 8 * (c.ne_eq > 0)
                   + 0.12 * max(c.age - 50, 0) + noise)
            sep_noise = _ar(r, n, 1.0, 0.9)
            fall = (20 + 0.5 * max(c.age - 40, 0) + 25 * c.delirium - 18 * (c.level == 4)
                    + 10 * (c.sed_depth < 0.3) * (c.level == 4) + _ar(r, n, 5.0, 0.9))
            for i in range(1, last):
                ts = at(admit, i + r.uniform(-0.2, 0.2))
                if ts >= discharge:
                    break
                if r.random() > 0.03:  # occasional missed calculation
                    mid = _version("epic_deterioration_index", ts)
                    if mid:
                        rows.append((mid, hid, ts, float(np.clip(round(edi[i], 1), 0, 100))))
                if i % 4 == 0:
                    mid = _version("epic_sepsis_model", ts)
                    if mid == "ESM-001":
                        v = 4 + 35 * c.infection[i] + 30 * min(c.severity[i], 1.2) + 8 * sep_noise[i]
                    else:
                        v = 2 + 55 * c.infection[i] * (1 if c.infected else 0.4) \
                            + 12 * min(c.severity[i], 1.2) + 4 * sep_noise[i]
                    if mid:
                        rows.append((mid, hid, ts, float(np.clip(round(v, 1), 0, 100))))
                if i % 12 == 6:
                    mid = _version("epic_fall_risk", ts)
                    if mid:
                        rows.append((mid, hid, ts, float(np.clip(round(fall[i], 0), 0, 100))))
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
        return df
