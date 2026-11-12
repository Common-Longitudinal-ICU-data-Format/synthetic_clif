"""CLIF 3.0 ``patient_attributes`` table.

Social history, function and social needs, charted at each admission.

Values are patient traits drawn from the patient-level seed, so a readmitted
patient keeps them (a smoker may quit; function can decline after a discharge
to a facility). They follow the rest of the dataset: patients with COPD are
mostly current or former smokers, employment and marital status follow age,
and functional status worsens with age and frailty.

Charting follows practice: the nursing admission history (tobacco, alcohol,
drugs, function, pregnancy for women of child-bearing age) at most
admissions; demographics at registration; organ-donor status once; and a
health-related social needs screen whose use rises from about 10% of
admissions in 2018 to most admissions by 2024.

Every ``attribute_value_category`` is one the mCIDE allows for that
``attribute_category`` (``ALLOWED_VALUES``, from
``clif_patient_attributes_attribute_value_categories.csv``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import stream, utc
from synthetic_clif.physiology.course import patient_seed_for_row, patient_traits

# mCIDE: attribute_category -> attribute_group
GROUP = {
    "alcohol_use": "substance_use_behavior",
    "employment_status": "sociodemographic",
    "financial_strain": "health_related_social_needs",
    "food_insecurity": "health_related_social_needs",
    "functional_status": "functional",
    "housing_status": "health_related_social_needs",
    "interpersonal_safety": "health_related_social_needs",
    "marital_status": "sociodemographic",
    "organ_donor": "care_preferences",
    "pregnancy_status": "reproductive",
    "smoking_status": "substance_use_behavior",
    "social_isolation": "health_related_social_needs",
    "substance_use": "substance_use_behavior",
    "transportation_barrier": "health_related_social_needs",
    "utility_insecurity": "health_related_social_needs",
}

# mCIDE: attribute_value_category values allowed for each attribute_category
# ("unknown" is allowed for all)
ALLOWED_VALUES = {
    "alcohol_use": {"none", "social", "heavy"},
    "employment_status": {"full_time", "part_time", "unemployed", "retired", "disabled", "student"},
    "financial_strain": {"none", "present"},
    "food_insecurity": {"secure", "insecure"},
    "functional_status": {"independent", "partially_dependent", "fully_dependent"},
    "housing_status": {"housed", "unstable", "undomiciled"},
    "interpersonal_safety": {"safe", "unsafe"},
    "marital_status": {"single", "married", "divorced", "separated", "widowed"},
    "organ_donor": {"yes", "no"},
    "pregnancy_status": {"pregnant", "not_pregnant", "postpartum"},
    "smoking_status": {"current", "former", "never"},
    "social_isolation": {"connected", "isolated"},
    "substance_use": {"none", "active", "history"},
    "transportation_barrier": {"none", "present"},
    "utility_insecurity": {"secure", "insecure"},
}
ALLOWED_VALUES = {k: v | {"unknown"} for k, v in ALLOWED_VALUES.items()}

NAMES = {
    "smoking_status": "TOBACCO USE",
    "alcohol_use": "ALCOHOL USE",
    "substance_use": "ILLICIT DRUG USE",
    "functional_status": "ADL STATUS PRIOR TO ADMISSION",
    "pregnancy_status": "PREGNANCY STATUS",
    "marital_status": "MARITAL STATUS",
    "employment_status": "EMPLOYMENT STATUS",
    "organ_donor": "ORGAN DONOR",
    "housing_status": "SDOH HOUSING STABILITY",
    "food_insecurity": "SDOH HUNGER VITAL SIGN",
    "transportation_barrier": "SDOH TRANSPORTATION NEEDS",
    "utility_insecurity": "SDOH UTILITIES",
    "financial_strain": "SDOH FINANCIAL RESOURCE STRAIN",
    "interpersonal_safety": "SDOH INTIMATE PARTNER VIOLENCE",
    "social_isolation": "SDOH SOCIAL CONNECTIONS",
}

VALUE_NAMES = {
    "current": "Current Every Day Smoker", "former": "Former Smoker", "never": "Never Smoker",
    "none": "None", "social": "Social Drinker", "heavy": "Heavy Drinker", "active": "Current Use",
    "history": "Past Use", "independent": "Independent", "partially_dependent": "Needs Assistance",
    "fully_dependent": "Dependent", "pregnant": "Pregnant", "not_pregnant": "Not Pregnant",
    "postpartum": "Postpartum", "single": "Single", "married": "Married", "divorced": "Divorced",
    "separated": "Legally Separated", "widowed": "Widowed", "full_time": "Full Time",
    "part_time": "Part Time", "unemployed": "Not Employed", "retired": "Retired",
    "disabled": "Disabled", "student": "Student", "yes": "Yes", "no": "No",
    "housed": "I have a steady place to live", "unstable": "Worried about losing housing",
    "undomiciled": "I do not have a steady place to live", "secure": "Never true",
    "insecure": "Sometimes true", "present": "Yes", "safe": "No", "unsafe": "Yes",
    "connected": "More than 3 times a week", "isolated": "Less than once a week",
    "unknown": "Patient unable to answer",
}

HRSN = ["housing_status", "food_insecurity", "transportation_barrier", "utility_insecurity",
        "financial_strain", "interpersonal_safety", "social_isolation"]


def _pick(r, options: dict) -> str:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return str(r.choice(keys, p=p / p.sum()))


def patient_profile(patient_seed: int, sex: str, age: float) -> dict:
    """Baseline attribute values for one patient (stable across stays)."""
    r = stream("attributes", patient_seed)
    t = patient_traits(patient_seed, sex)
    v = {}
    if t["copd"]:
        v["smoking_status"] = _pick(r, {"current": 0.4, "former": 0.55, "never": 0.05})
    else:
        v["smoking_status"] = _pick(r, {"current": 0.15, "former": 0.25, "never": 0.56, "unknown": 0.04})
    heavy_drinker = r.random() < 0.09
    v["alcohol_use"] = "heavy" if heavy_drinker else _pick(r, {"none": 0.55, "social": 0.42, "unknown": 0.03})
    v["substance_use"] = _pick(r, {"none": 0.86, "active": 0.05 + 0.08 * heavy_drinker,
                                   "history": 0.06, "unknown": 0.03})
    if age < 30:
        marital = {"single": 0.75, "married": 0.22, "divorced": 0.02, "separated": 0.01}
    elif age < 65:
        marital = {"single": 0.25, "married": 0.5, "divorced": 0.17, "separated": 0.04, "widowed": 0.04}
    else:
        marital = {"single": 0.08, "married": 0.47, "divorced": 0.15, "separated": 0.02,
                   "widowed": 0.18 + 0.006 * (age - 65)}
    v["marital_status"] = _pick(r, marital)
    if age >= 67:
        emp = {"retired": 0.85, "part_time": 0.05, "full_time": 0.04, "disabled": 0.06}
    elif age < 25:
        emp = {"student": 0.35, "part_time": 0.25, "full_time": 0.2, "unemployed": 0.2}
    else:
        emp = {"full_time": 0.45, "part_time": 0.1, "unemployed": 0.13, "disabled": 0.17 + 0.1 * t["ckd"],
               "retired": 0.05 + 0.01 * max(age - 55, 0), "unknown": 0.03}
    v["employment_status"] = _pick(r, emp)
    frail = (age - 50) / 40 + 0.4 * t["copd"] + 0.4 * t["ckd"] + r.normal(0, 0.4)
    v["functional_status"] = ("fully_dependent" if frail > 1.3 else
                              "partially_dependent" if frail > 0.7 else "independent")
    v["organ_donor"] = _pick(r, {"yes": 0.45, "no": 0.4, "unknown": 0.15})
    # social needs cluster together
    need = r.normal(0, 1) + 0.8 * (v["employment_status"] in ("unemployed", "disabled")) \
        + 0.8 * (v["substance_use"] == "active")
    v["housing_status"] = ("undomiciled" if need > 2.3 else "unstable" if need > 1.8 else
                           _pick(r, {"housed": 0.97, "unknown": 0.03}))
    v["food_insecurity"] = "insecure" if need + r.normal(0, 0.5) > 1.4 else "secure"
    v["financial_strain"] = "present" if need + r.normal(0, 0.5) > 1.0 else "none"
    v["transportation_barrier"] = "present" if need + r.normal(0, 0.6) > 1.5 else "none"
    v["utility_insecurity"] = "insecure" if need + r.normal(0, 0.6) > 1.7 else "secure"
    v["interpersonal_safety"] = "unsafe" if r.random() < 0.02 else "safe"
    lonely = r.normal(0, 1) + 0.9 * (v["marital_status"] in ("widowed", "divorced", "separated")) \
        + 0.02 * max(age - 60, 0)
    v["social_isolation"] = "isolated" if lonely > 1.4 else "connected"
    return v


class PatientAttributesGenerator(BaseGenerator):
    COLUMNS = ["patient_id", "recorded_dttm", "attribute_name", "attribute_category",
               "attribute_group", "attribute_value_name", "attribute_value_category"]

    def generate(self, patients_df: pd.DataFrame, hospitalizations_df: pd.DataFrame) -> pd.DataFrame:
        sex = {}
        if "sex_category" in patients_df.columns:
            sex = dict(zip(patients_df["patient_id"], patients_df["sex_category"].astype(str).str.lower()))
        hosp = hospitalizations_df.dropna(subset=["admission_dttm"]).sort_values(
            ["patient_id", "admission_dttm"])
        rows = []
        for pid, stays in hosp.groupby("patient_id", sort=False):
            first = stays.iloc[0]
            pseed = patient_seed_for_row(first)
            if pseed is None:
                continue
            age0 = first.get("age_at_admission")
            age0 = 60.0 if age0 is None or pd.isna(age0) else float(age0)
            s = sex.get(pid, str(first.get("_sex") or "")).lower()
            prof = patient_profile(pseed, s, age0)
            r = stream("attributes_charting", pseed)
            prev_dispo = None
            for k, (_, st) in enumerate(stays.iterrows()):
                admit = utc(st["admission_dttm"])
                dc = st.get("discharge_dttm")
                dc = utc(dc) if dc is not None and not pd.isna(dc) else admit + pd.Timedelta(5, unit="D")
                age = st.get("age_at_admission")
                age = age0 if age is None or pd.isna(age) else float(age)
                year = admit.year
                vals = dict(prof)
                if k and vals["smoking_status"] == "current" and r.random() < 0.15:
                    vals["smoking_status"] = prof["smoking_status"] = "former"
                if prev_dispo in ("skilled nursing facility (snf)", "long term care hospital (ltach)",
                                  "acute inpatient rehab facility") \
                        and prof["functional_status"] != "fully_dependent":
                    prof["functional_status"] = ("partially_dependent"
                                                 if prof["functional_status"] == "independent"
                                                 else "fully_dependent")
                    vals["functional_status"] = prof["functional_status"]
                prev_dispo = str(st.get("discharge_category") or "").strip().lower()

                def add(cat, h):
                    val = vals[cat]
                    if r.random() < 0.03:
                        val = "unknown"
                    rows.append({
                        "patient_id": pid,
                        "recorded_dttm": min(admit + pd.Timedelta(float(h), unit="h"),
                                             dc - pd.Timedelta(float(r.uniform(5, 60)), unit="min")),
                        "attribute_name": NAMES[cat],
                        "attribute_category": cat,
                        "attribute_group": GROUP[cat],
                        "attribute_value_name": VALUE_NAMES[val],
                        "attribute_value_category": val,
                    })

                # registration
                if k == 0 or r.random() < 0.3:
                    add("marital_status", r.uniform(-1, 0.5))
                    add("employment_status", r.uniform(-1, 0.5))
                if k == 0:
                    add("organ_donor", r.uniform(-1, 0.5))
                # nursing admission history
                if r.random() < 0.9:
                    h = r.uniform(1, 12)
                    for cat in ("smoking_status", "alcohol_use", "substance_use", "functional_status"):
                        if r.random() < 0.95:
                            add(cat, h + r.uniform(0, 0.2))
                    if s == "female" and 12 <= age <= 50:
                        preg = _pick(r, {"not_pregnant": 0.95, "pregnant": 0.035, "postpartum": 0.015})
                        vals["pregnancy_status"] = preg
                        add("pregnancy_status", h + r.uniform(0, 0.2))
                # social needs screen, adopted over the study years
                if r.random() < float(np.clip(0.1 + 0.1 * (year - 2018), 0.1, 0.75)):
                    h = r.uniform(4, 36)
                    for cat in HRSN:
                        add(cat, h + r.uniform(0, 0.1))
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)
        return df
