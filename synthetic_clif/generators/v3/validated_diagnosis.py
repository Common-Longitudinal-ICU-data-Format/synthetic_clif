"""CLIF 3.0 ``validated_diagnosis`` table (clinician-adjudicated labels).

A review cohort is drawn as a study would draw one: about half of stays with
ICU-level support (invasive or non-invasive ventilation, vasopressors,
dialysis) and a few others. For each reviewed stay, reviewers label every
condition with a signal in the chart, using the same course that drives the
billed ``hospital_diagnosis`` codes:

- **sepsis** (Sepsis-3): suspected infection with organ dysfunction. Starts
  when infection was recognised. Suspected infection that was not one, or
  infection without organ dysfunction, is ``ruled_out``.
- **arf** (acute respiratory failure): non-invasive or invasive ventilation;
  high-flow oxygen alone is ``ruled_out``.
- **ards** (Berlin): PaO2/FiO2 under 300 on invasive ventilation with PEEP of
  at least 5 and bilateral opacities (the course's respiratory score high
  enough that the chest film reports them) for 12 hours.
  Transient hypoxaemia is ``ruled_out``; meeting criteria in cardiogenic
  shock is ``uncertain`` (hydrostatic oedema cannot be excluded).
- **aki** (KDIGO): kidney injury by creatinine, reviewed after an automated
  screen; mild rises are ``ruled_out``.

About 5% of confirmed labels are instead ``uncertain`` and go to a consensus
panel. Reviews are finalised one to twelve months after discharge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import at, critical, iter_courses, stream

NAMES = {
    "sepsis": "Sepsis (Sepsis-3)",
    "arf": "Acute respiratory failure",
    "ards": "ARDS (Berlin definition)",
    "aki": "Acute kidney injury (KDIGO)",
}
REVIEWERS = [f"REV-{i:02d}" for i in range(1, 9)]


def _sustained(mask, k):
    out = np.zeros(len(mask), dtype=bool)
    streak = 0
    for i, v in enumerate(mask):
        streak = streak + 1 if v else 0
        out[i] = streak >= k
    return out


def _onset(mask):
    idx = np.nonzero(mask)[0]
    return float(idx[0]) if len(idx) else None


def _offset(mask):
    idx = np.nonzero(mask)[0]
    return float(idx[-1] + 1) if len(idx) else None


def course_labels(c) -> list[tuple[str, str, float, float | None]]:
    """[(category, status, start_h, end_h)] for the conditions a reviewer would weigh."""
    out = []
    n = c.n_hours
    if c.infection_recognized_h is not None:
        h0 = c.infection_recognized_h
        i0 = int(min(h0, n - 1))
        after = slice(i0, n)
        dysfunction = ((c.shock[after] > 0.3) | (c.level[after] >= 3) | (c.aki[after] > 0.4)).any()
        if c.infected and dysfunction:
            end = max((a["end"] for a in c.antibiotics), default=None)
            out.append(("sepsis", "confirmed", h0, end))
        else:
            out.append(("sepsis", "ruled_out", h0, None))
    vent = c.level >= 3
    if vent.any():
        out.append(("arf", "confirmed", _onset(vent), _offset(vent)))
    elif (c.level == 2).any():
        out.append(("arf", "ruled_out", _onset(c.level == 2), None))
    low = (c.level == 4) & (c.pf_true < 300)
    if low.any():
        # bilateral opacities on the chest film (the radiology report's
        # "diffuse bilateral airspace opacities") on PEEP >= 5
        peep = np.nan_to_num(c.peep, nan=0.0)
        met = _sustained(low & (c.resp > 0.65) & (peep >= 5), 12)
        if met.any():
            status = "uncertain" if c.shock_type == "cardiogenic" else "confirmed"
            out.append(("ards", status, _onset(met) - 11, _offset(low)))
        else:
            out.append(("ards", "ruled_out", _onset(low), None))
    if (c.aki > 0.4).any():
        out.append(("aki", "confirmed", _onset(c.aki > 0.4), _offset(c.aki > 0.4)))
    elif (c.aki > 0.2).any():
        out.append(("aki", "ruled_out", _onset(c.aki > 0.2), None))
    return out


class ValidatedDiagnosisGenerator(BaseGenerator):
    COLUMNS = ["hospitalization_id", "diagnosis_name", "diagnosis_category", "diagnosis_start_dttm",
               "diagnosis_end_dttm", "validation_method", "diagnosis_status", "reviewer_id",
               "review_timestamp"]

    def generate(self, hospitalizations_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row, c, admit, discharge in iter_courses(hospitalizations_df):
            r = stream("validated_dx", c.rng_seed)
            icu_level = critical(c).any() or bool(c.ihd_sessions)
            if r.random() > (0.5 if icu_level else 0.05):
                continue
            labels = course_labels(c)
            if not labels:
                continue
            reviewer = str(r.choice(REVIEWERS))
            reviewed = discharge + pd.Timedelta(float(r.uniform(30, 365)), unit="D")
            for cat, status, s, e in labels:
                if status == "confirmed" and r.random() < 0.05:
                    status = "uncertain"
                if status == "uncertain":
                    method = "consensus_panel"
                elif cat == "aki":
                    method = "automated_with_review"
                else:
                    method = "manual_chart_review" if r.random() < 0.9 else "consensus_panel"
                start = min(max(at(admit, s), admit), discharge)
                end = None
                if e is not None and status != "ruled_out":
                    end = at(admit, e)
                    end = None if end >= discharge else max(end, start)
                rows.append({
                    "hospitalization_id": row["hospitalization_id"],
                    "diagnosis_name": NAMES[cat],
                    "diagnosis_category": cat,
                    "diagnosis_start_dttm": start,
                    "diagnosis_end_dttm": end,
                    "validation_method": method,
                    "diagnosis_status": status,
                    "reviewer_id": reviewer,
                    "review_timestamp": reviewed + pd.Timedelta(float(r.uniform(0, 240)), unit="min"),
                })
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        for col in ("diagnosis_start_dttm", "diagnosis_end_dttm", "review_timestamp"):
            df[col] = pd.to_datetime(df[col], utc=True)
        return df
