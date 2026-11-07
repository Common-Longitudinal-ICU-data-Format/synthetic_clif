"""Patient diagnosis (problem list/history) generator.

Chronic conditions belong to the patient. Diabetes, COPD, chronic kidney
disease and obesity come from the same patient-level traits that shape the
hospital course (baseline creatinine, CO2 retention, insulin, weight), so the
problem list agrees with the labs and vitals. The other conditions cluster
the way they do clinically (hypertension with diabetes and CKD, heart
failure with coronary disease and atrial fibrillation) and grow more common
with age. Onset dates fall before the patient's first admission.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import _stable_hash, patient_seed_for_row, patient_traits

CODES = {
    "Hypertension": "I10",
    "Type 2 Diabetes": "E11.9",
    "Hyperlipidemia": "E78.5",
    "Coronary Artery Disease": "I25.10",
    "Heart Failure": "I50.9",
    "Atrial Fibrillation": "I48.91",
    "COPD": "J44.9",
    "Asthma": "J45.909",
    "Chronic Kidney Disease": "N18.9",
    "Obesity": "E66.9",
    "Depression": "F32.9",
    "Anxiety": "F41.9",
    "Hypothyroidism": "E03.9",
    "GERD": "K21.0",
    "Osteoarthritis": "M19.90",
}


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


class PatientDiagnosisGenerator(BaseGenerator):
    """Generate the patient-level problem list from patient traits."""

    COLUMNS = ["patient_id", "hospitalization_id", "diagnosis_code", "diagnosis_code_format",
               "start_dttm", "end_dttm", "source_type"]

    def generate(self, patients_df: pd.DataFrame, hospitalizations_df: pd.DataFrame = None) -> pd.DataFrame:
        first = {}
        if hospitalizations_df is not None and len(hospitalizations_df) > 0:
            h = hospitalizations_df.sort_values("admission_dttm")
            first = {pid: g.iloc[0] for pid, g in h.groupby("patient_id", sort=False)}

        records = []
        for _, patient in patients_df.iterrows():
            pid = patient["patient_id"]
            stay = first.get(pid)
            records.extend(self._patient(patient, stay))

        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["start_dttm"] = pd.to_datetime(df["start_dttm"], utc=True)
            df["end_dttm"] = pd.to_datetime(df["end_dttm"], utc=True)
        return df

    def _patient(self, patient, stay) -> list[dict]:
        pid = patient["patient_id"]
        row = {"patient_id": pid}
        if stay is not None and "_patient_seed" in stay.index:
            row["_patient_seed"] = stay["_patient_seed"]
        pseed = patient_seed_for_row(row)
        sex = str(patient.get("sex_category") or "").strip().lower()
        if sex not in ("male", "female"):
            # unknown sex: a fixed draw from the patient seed (only height depends on it)
            sex = "male" if np.random.default_rng(pseed).random() < 0.55 else "female"
        t = patient_traits(pseed, sex)
        r = np.random.default_rng((pseed + _stable_hash("problem_list")) & 0xFFFFFFFF)

        anchor = None
        if stay is not None and not pd.isna(stay.get("admission_dttm")):
            anchor = pd.Timestamp(stay["admission_dttm"])
        age = 60.0
        if stay is not None and not pd.isna(stay.get("age_at_admission")):
            age = float(stay["age_at_admission"])
        elif anchor is not None and not pd.isna(patient.get("birth_date")):
            age = (anchor.tz_localize(None) - pd.Timestamp(patient["birth_date"])).days / 365.25
        a = (age - 60) / 10

        has = {
            "Type 2 Diabetes": t["diabetic"],
            "COPD": t["copd"],
            "Chronic Kidney Disease": t["ckd"],
            "Obesity": t["bmi"] >= 30,
        }
        p = {
            "Hypertension": _logistic(-0.3 + 0.5 * a + 1.0 * has["Type 2 Diabetes"] + 1.2 * has["Chronic Kidney Disease"]
                                      + 0.6 * has["Obesity"]),
            "Hyperlipidemia": _logistic(-0.7 + 0.3 * a + 0.9 * has["Type 2 Diabetes"]),
            "Asthma": 0.08 + 0.04 * has["Obesity"],
            "Depression": 0.15,
            "Anxiety": 0.12,
            "Hypothyroidism": 0.06 + 0.06 * (sex == "female"),
            "GERD": 0.15 + 0.08 * has["Obesity"],
            "Osteoarthritis": _logistic(-2.0 + 0.6 * a + 0.5 * has["Obesity"]),
        }
        for k, v in p.items():
            has[k] = bool(r.random() < v)
        has["Coronary Artery Disease"] = bool(r.random() < _logistic(
            -2.4 + 0.6 * a + 0.8 * has["Type 2 Diabetes"] + 0.6 * has["Hypertension"]
            + 0.5 * has["Hyperlipidemia"] + 0.4 * (sex == "male")))
        has["Atrial Fibrillation"] = bool(r.random() < _logistic(-2.8 + 0.7 * a + 0.4 * has["Hypertension"]))
        has["Heart Failure"] = bool(r.random() < _logistic(
            -3.0 + 0.5 * a + 1.2 * has["Coronary Artery Disease"] + 0.9 * has["Atrial Fibrillation"]
            + 0.6 * has["Chronic Kidney Disease"]))

        hosp_id = stay["hospitalization_id"] if stay is not None else None
        rows = []
        for name in CODES:
            if not has[name]:
                continue
            start = None
            if anchor is not None:
                years = r.uniform(0.3, max(min(20.0, age - 18.0), 0.5))
                start = anchor - timedelta(days=float(years) * 365.25)
            rows.append({
                "patient_id": pid,
                "hospitalization_id": hosp_id,
                "diagnosis_code": CODES[name],
                "diagnosis_code_format": "ICD-10-CM",
                "start_dttm": start,
                "end_dttm": None,
                "source_type": str(r.choice(["Problem List", "Medical History", "Encounter"], p=[0.5, 0.3, 0.2])),
            })
        return rows
