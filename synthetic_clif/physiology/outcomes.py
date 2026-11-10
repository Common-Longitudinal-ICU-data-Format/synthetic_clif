"""Outcomes that follow from the hospital course.

Who dies in hospital, where survivors go, and who dies after discharge used
to be drawn at random, independently of how sick anyone was. Here they are
functions of the course: peak illness, organ support, and age.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from synthetic_clif.physiology.course import HospitalCourse, _stable_hash

DISPOSITIONS = {
    "Home": 0.55,
    "Skilled Nursing Facility (SNF)": 0.12,
    "Acute Inpatient Rehab Facility": 0.05,
    "Hospice": 0.02,
    "Long Term Care Hospital (LTACH)": 0.02,
    "Acute Care Hospital": 0.02,
    "Group Home": 0.01,
    "Chemical Dependency": 0.005,
    "Against Medical Advice (AMA)": 0.02,
    "Assisted Living": 0.03,
    "Other": 0.02,
    "Psychiatric Hospital": 0.005,
    "Shelter": 0.005,
    "Jail": 0.003,
}


def outcome_rng(course_seed: int) -> np.random.Generator:
    return np.random.default_rng((int(course_seed) + _stable_hash("outcome")) & 0xFFFFFFFF)


def death_probability(c: HospitalCourse) -> float:
    """In-hospital mortality risk for a course built without a terminal decline."""
    peak = float(min(c.severity.max(), 1.4))
    imv = bool((c.level == 4).any())
    pressor = bool((c.ne_eq > 0).any())
    high_pressor = bool((c.ne_eq > 0.3).any())
    crrt = bool(c.crrt.any())
    mcs = bool(c.mcs)
    logit = (-5.3 + 2.6 * peak + 0.7 * imv + 0.6 * pressor + 0.9 * high_pressor
             + 0.6 * crrt + 0.8 * mcs + 0.035 * (c.age - 65))
    return 1.0 / (1.0 + math.exp(-logit))


def discharge_disposition(c: HospitalCourse, rng, p_death: float) -> str:
    """Where a survivor goes, given what the stay took out of them."""
    w = dict(DISPOSITIONS)
    los_days = c.n_hours / 24.0
    imv_h = int((c.level == 4).sum())
    trach = bool(c.trach.any())
    if c.age >= 75:
        w["Skilled Nursing Facility (SNF)"] *= 3
        w["Assisted Living"] *= 2
        w["Home"] *= 0.6
        w["Against Medical Advice (AMA)"] *= 0.2
    if imv_h > 72:
        w["Skilled Nursing Facility (SNF)"] *= 2
        w["Acute Inpatient Rehab Facility"] *= 2.5
        w["Home"] *= 0.6
    if trach:
        w["Long Term Care Hospital (LTACH)"] += 0.6
        w["Home"] *= 0.2
    if los_days > 14:
        w["Skilled Nursing Facility (SNF)"] *= 1.5
    if p_death > 0.3 and c.age >= 70:
        w["Hospice"] *= 6
    if c.acuity == 1 and c.age < 50:
        w["Against Medical Advice (AMA)"] *= 2
    cats = list(w)
    p = np.array([w[k] for k in cats])
    return str(rng.choice(cats, p=p / p.sum()))


def days_to_death_after_discharge(c: HospitalCourse, rng, disposition: str,
                                  p_death: float) -> Optional[float]:
    """Days from discharge to death within a year, or None for a survivor."""
    if disposition == "Hospice":
        return float(rng.uniform(1, 30)) if rng.random() < 0.85 else None
    p = 0.05 + 0.5 * p_death + 0.008 * max(c.age - 65, 0)
    if disposition in ("Skilled Nursing Facility (SNF)", "Long Term Care Hospital (LTACH)"):
        p += 0.1
    if rng.random() < min(p, 0.8):
        return float(rng.uniform(7, 365))
    return None
