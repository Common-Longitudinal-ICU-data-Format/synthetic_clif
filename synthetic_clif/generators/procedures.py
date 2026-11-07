"""Patient procedures and hospital diagnosis generators."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import course_for_row


class PatientProceduresGenerator(BaseGenerator):
    """Generate synthetic patient procedure data.

    Creates patient_procedures table with:
    - ICU-relevant procedures (central lines, intubations, etc.)
    - ICD-10-PCS and CPT codes
    """

    # Common ICU procedures with codes
    PROCEDURES = {
        "Central Line Insertion": {
            "icd10_pcs": ["02HV33Z", "02HV43Z"],
            "cpt": ["36556", "36558"],
            "probability": 0.4,
        },
        "Arterial Line Insertion": {
            "icd10_pcs": ["02H633Z", "02H733Z"],
            "cpt": ["36620"],
            "probability": 0.35,
        },
        "Intubation": {
            "icd10_pcs": ["0BH17EZ"],
            "cpt": ["31500"],
            "probability": 0.15,
        },
        "Extubation": {
            "icd10_pcs": ["0BH18EZ"],
            "cpt": ["31500"],
            "probability": 0.12,
        },
        "Bronchoscopy": {
            "icd10_pcs": ["0BJ08ZZ"],
            "cpt": ["31622", "31623"],
            "probability": 0.08,
        },
        "Chest Tube Insertion": {
            "icd10_pcs": ["0W9930Z", "0W9940Z"],
            "cpt": ["32551"],
            "probability": 0.06,
        },
        "Paracentesis": {
            "icd10_pcs": ["0W9G30Z"],
            "cpt": ["49083"],
            "probability": 0.05,
        },
        "Thoracentesis": {
            "icd10_pcs": ["0W9B30Z"],
            "cpt": ["32555"],
            "probability": 0.04,
        },
        "Lumbar Puncture": {
            "icd10_pcs": ["009U3ZX"],
            "cpt": ["62270"],
            "probability": 0.03,
        },
        "Dialysis Catheter Insertion": {
            "icd10_pcs": ["02HV33Z"],
            "cpt": ["36558"],
            "probability": 0.08,
        },
        "PICC Line Insertion": {
            "icd10_pcs": ["02HV33Z"],
            "cpt": ["36569"],
            "probability": 0.15,
        },
        "Tracheostomy": {
            "icd10_pcs": ["0B110F4"],
            "cpt": ["31600"],
            "probability": 0.03,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate patient procedures.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with patient_procedures columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            course = course_for_row(hosp)
            hosp_procedures = self._course_procedures(
                hosp_id, admit_time, discharge_time, course,
                str(hosp.get("admission_type_category") or ""),
            )
            records.extend(hosp_procedures)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["procedure_billed_dttm"] = pd.to_datetime(df["procedure_billed_dttm"], utc=True)

        return df

    # (ICD-10-PCS, CPT) per course event
    COURSE_CODES = {
        "intubation": ("0BH17EZ", "31500"),
        "art_line": ("03HY32Z", "36620"),
        "central_line": ("02HV33Z", "36556"),
        "dialysis_line": ("05HY33Z", "36556"),
        "pa_catheter": ("02HP32Z", "93503"),
        "tracheostomy": ("0B110F4", "31600"),
        "crrt": ("5A1D90Z", "90945"),
        "VV_ECMO": ("5A1522H", "33946"),
        "VA_ECMO": ("5A1522G", "33946"),
        "Impella_CP": ("02HA3RZ", "33990"),
        "Impella_5.5": ("02HA0RZ", "33990"),
        "IABP": ("5A02210", "33967"),
        "Packed RBCs": ("30233N1", "36430"),
        "Platelets": ("30233R1", "36430"),
        "Fresh Frozen Plasma": ("30233L1", "36430"),
        "Cryoprecipitate": ("30233M1", "36430"),
        "bal": ("0BJ08ZZ", "31624"),
    }
    # Mechanical ventilation by duration (ICD-10-PCS only)
    VENT_DURATION = [(24, "5A1935Z"), (96, "5A1945Z"), (float("inf"), "5A1955Z")]

    def _course_procedures(self, hosp_id, admit_time, discharge_time, course, admission_type) -> list[dict]:
        """Procedures at the times the course performed them."""
        if course is None:
            return self._generate_hospitalization_procedures(hosp_id, admit_time, discharge_time)
        events = []  # (hour, key or (pcs, cpt))
        n = course.n_hours

        def starts(mask):
            return [i for i in range(n) if mask[i] and (i == 0 or not mask[i - 1])]

        for s, e in course.imv_episodes:
            if s > 0 or admission_type not in ("osh", "facility"):
                events.append((s, "intubation"))
            hours = e - s
            code = next(c for lim, c in self.VENT_DURATION if hours <= lim)
            events.append((s, (code, None)))
        trach = np.nonzero(course.trach)[0]
        if len(trach):
            events.append((float(trach[0]) - 0.5, "tracheostomy"))
        for name in ("art_line", "central_line", "dialysis_line", "pa_catheter"):
            for i in starts(getattr(course, name)):
                events.append((i, name))
        crrt = starts(course.crrt)
        for i in crrt:
            events.append((i, "crrt"))
        for m in course.mcs:
            events.append((m["start"], m["device"]))
        for tr in course.transfusions:
            events.append((tr["h"], tr["component"]))
        for cx in course.cultures:
            if cx["fluid"] == "respiratory" and course.level[min(int(cx["h"]), n - 1)] == 4 \
                    and self.rng.random() < 0.3:
                events.append((cx["h"] - 0.5, "bal"))
        # Bedside procedures not modelled by the course
        for pcs, cpt, p in (("0W9930Z", "32551", 0.04), ("0W9G30Z", "49083", 0.04),
                            ("0W9B30Z", "32555", 0.03), ("009U3ZX", "62270", 0.02)):
            if self.rng.random() < p:
                events.append((self.rng.uniform(0, n * 0.8), (pcs, cpt)))

        rows = []
        for h, key in sorted(events, key=lambda x: x[0]):
            pcs, cpt = self.COURSE_CODES[key] if isinstance(key, str) else key
            use_pcs = cpt is None or self.rng.random() < 0.7
            ts = admit_time + timedelta(hours=float(max(h, 0)) + self.rng.uniform(0, 0.5))
            if ts > discharge_time:
                continue
            rows.append({
                "hospitalization_id": hosp_id,
                "procedure_billed_dttm": ts,
                "procedure_code": pcs if use_pcs else cpt,
                "procedure_code_format": "ICD10PCS" if use_pcs else "CPT",
                "billing_provider_id": f"PROV-{self.rng.integers(10000, 99999)}",
                "performing_provider_id": f"PROV-{self.rng.integers(10000, 99999)}",
            })
        return rows

    def _generate_hospitalization_procedures(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate procedures for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        for proc_name, proc_data in self.PROCEDURES.items():
            if self.rng.random() < proc_data["probability"]:
                # Procedure timing
                if proc_name in ["Intubation", "Central Line Insertion", "Arterial Line Insertion"]:
                    # Early procedures
                    hours_from_admit = self.rng.uniform(0, min(12, los_hours))
                elif proc_name in ["Extubation", "Tracheostomy"]:
                    # Later procedures
                    hours_from_admit = self.rng.uniform(
                        min(24, los_hours * 0.3), los_hours * 0.9
                    )
                else:
                    hours_from_admit = self.rng.uniform(0, los_hours * 0.8)

                proc_time = admit_time + timedelta(hours=hours_from_admit)

                # Select code type
                use_icd10 = self.rng.random() < 0.7
                if use_icd10:
                    code = self.rng.choice(proc_data["icd10_pcs"])
                    code_type = "ICD10PCS"
                else:
                    code = self.rng.choice(proc_data["cpt"])
                    code_type = "CPT"

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "procedure_billed_dttm": proc_time,
                        "procedure_code": code,
                        "procedure_code_format": code_type,
                        "billing_provider_id": f"PROV-{self.rng.integers(10000, 99999)}",
                        "performing_provider_id": f"PROV-{self.rng.integers(10000, 99999)}",
                    }
                )

        return records


class HospitalDiagnosisGenerator(BaseGenerator):
    """Generate synthetic hospital diagnosis data.

    Creates hospital_diagnosis table with:
    - ICD-10-CM diagnosis codes
    - Principal and secondary diagnoses
    - POA (present on admission) flags
    """

    # Common ICU diagnoses with ICD-10-CM codes
    DIAGNOSES = {
        # Respiratory
        "Acute respiratory failure": {
            "codes": ["J96.00", "J96.01", "J96.02"],
            "probability": 0.35,
            "is_principal": True,
        },
        "Pneumonia": {
            "codes": ["J18.9", "J15.9", "J13"],
            "probability": 0.25,
            "is_principal": True,
        },
        "ARDS": {
            "codes": ["J80"],
            "probability": 0.08,
            "is_principal": True,
        },
        "COPD exacerbation": {
            "codes": ["J44.1"],
            "probability": 0.12,
            "is_principal": True,
        },
        # Cardiovascular
        "Sepsis": {
            "codes": ["A41.9", "R65.20", "R65.21"],
            "probability": 0.30,
            "is_principal": True,
        },
        "Acute kidney injury": {
            "codes": ["N17.9", "N17.0", "N17.1"],
            "probability": 0.25,
            "is_principal": False,
        },
        "Acute MI": {
            "codes": ["I21.3", "I21.4", "I21.9"],
            "probability": 0.10,
            "is_principal": True,
        },
        "Heart failure": {
            "codes": ["I50.9", "I50.21", "I50.31"],
            "probability": 0.20,
            "is_principal": False,
        },
        "Atrial fibrillation": {
            "codes": ["I48.91", "I48.0", "I48.1"],
            "probability": 0.18,
            "is_principal": False,
        },
        # Neurologic
        "Stroke": {
            "codes": ["I63.9", "I61.9"],
            "probability": 0.08,
            "is_principal": True,
        },
        "Altered mental status": {
            "codes": ["R41.82"],
            "probability": 0.15,
            "is_principal": False,
        },
        # GI
        "GI bleed": {
            "codes": ["K92.2", "K92.0", "K92.1"],
            "probability": 0.08,
            "is_principal": True,
        },
        "Acute pancreatitis": {
            "codes": ["K85.9", "K85.0"],
            "probability": 0.05,
            "is_principal": True,
        },
        # Metabolic
        "DKA": {
            "codes": ["E11.10", "E10.10"],
            "probability": 0.05,
            "is_principal": True,
        },
        "Hyponatremia": {
            "codes": ["E87.1"],
            "probability": 0.12,
            "is_principal": False,
        },
        "Hyperkalemia": {
            "codes": ["E87.5"],
            "probability": 0.08,
            "is_principal": False,
        },
        # Comorbidities
        "Hypertension": {
            "codes": ["I10"],
            "probability": 0.50,
            "is_principal": False,
        },
        "Diabetes mellitus": {
            "codes": ["E11.9", "E10.9"],
            "probability": 0.35,
            "is_principal": False,
        },
        "Chronic kidney disease": {
            "codes": ["N18.3", "N18.4", "N18.5"],
            "probability": 0.20,
            "is_principal": False,
        },
        "Obesity": {
            "codes": ["E66.9", "E66.01"],
            "probability": 0.25,
            "is_principal": False,
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate hospital diagnoses.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with hospital_diagnosis columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            course = course_for_row(hosp)
            if course is None:
                hosp_diagnoses = self._generate_hospitalization_diagnoses(hosp_id)
            else:
                hosp_diagnoses = self._course_diagnoses(hosp_id, course)
            records.extend(hosp_diagnoses)

        return pd.DataFrame(records)

    INFECTION_CODES = {
        "pulmonary": "J18.9",
        "urinary": "N39.0",
        "bloodstream": "A41.9",
        "abdominal": "K65.9",
        "skin": "L03.90",
    }
    OTHER_PRINCIPALS = [("I63.9", 0.2), ("K85.9", 0.15), ("I21.4", 0.2), ("T40.2X1A", 0.15),
                        ("I50.21", 0.15), ("R41.82", 0.15)]

    def _course_diagnoses(self, hosp_id, course) -> list[dict]:
        """Diagnoses the course supports, coded present-on-admission when
        they began in the first day."""
        n = course.n_hours
        dx = []  # (code, onset_hour, principal_rank)

        def onset(mask):
            idx = np.nonzero(mask)[0]
            return float(idx[0]) if len(idx) else None

        shock_h = onset(course.ne_eq > 0)
        if course.infected:
            h = course.infection_recognized_h or 0.0
            src = self.INFECTION_CODES[course.infection_source or "pulmonary"]
            dx.append(("A41.9", h, 1))
            dx.append(("R65.21" if shock_h is not None else "R65.20", h, 9))
            if src != "A41.9":
                dx.append((src, h, 9))
        resp_h = onset(course.level >= 3)
        if resp_h is not None:
            dx.append(("J96.02" if course.lab_base.get("hypercapnic") else "J96.01", resp_h, 2))
            if course.lab_base.get("hypercapnic") and self.rng.random() < 0.7:
                dx.append(("J44.1", resp_h, 3))
        ards = (course.level == 4) & (course.pf_true < 200)
        if ards.sum() >= 24:
            dx.append(("J80", onset(ards), 9))
        if shock_h is not None and not course.infected:
            code = {"cardiogenic": "R57.0", "hypovolemic": "R57.1"}.get(course.shock_type, "R57.8")
            dx.append((code, shock_h, 2))
        aki_h = onset(course.aki > 0.4)
        if aki_h is not None:
            dx.append(("N17.0" if course.crrt.any() else "N17.9", aki_h, 9))
        if course.bleed.max() > 1.0:
            dx.append(("K92.2", onset(course.bleed > 0.5), 2))
        if course.delirium.any():
            dx.append(("G93.41" if self.rng.random() < 0.5 else "R41.0", onset(course.delirium), 9))
        if any(t["component"] == "Packed RBCs" for t in course.transfusions):
            dx.append(("D62", min(t["h"] for t in course.transfusions), 9))
        if any(t["component"] == "Platelets" for t in course.transfusions):
            dx.append(("D69.6", 0.0, 9))
        if course.shock_type == "cardiogenic":
            dx.append(("I50.21" if self.rng.random() < 0.6 else "I21.4", 0.0, 2))
        if course.expired and course.comfort_h is None and self.rng.random() < 0.5:
            dx.append(("I46.9", float(n - 2), 9))
        # Chronic conditions
        bmi = course.weight_kg / (course.height_cm / 100) ** 2
        if course.diabetic:
            dx.append(("E11.65" if course.insulin.any() else "E11.9", 0.0, 9))
        if self.rng.random() < 0.3 + 0.006 * max(course.age - 40, 0):
            dx.append(("I10", 0.0, 9))
        if bmi >= 30:
            dx.append(("E66.9" if bmi < 40 else "E66.01", 0.0, 9))
        if self.rng.random() < 0.1 + 0.004 * max(course.age - 50, 0):
            dx.append(("I48.91", 0.0, 9))
        if self.rng.random() < 0.12 + 0.3 * (course.lab_base["creatinine"] > 1.3):
            dx.append(("N18.3", 0.0, 9))
        if not any(r <= 3 for _, _, r in dx):
            codes, w = zip(*self.OTHER_PRINCIPALS)
            dx.append((str(self.rng.choice(codes, p=np.array(w) / sum(w))), 0.0, 4))

        seen, rows = set(), []
        principal = min(dx, key=lambda d: d[2])[0]
        for code, h, _ in dx:
            if code in seen:
                continue
            seen.add(code)
            rows.append({
                "hospitalization_id": hosp_id,
                "diagnosis_code": code,
                "diagnosis_code_format": "ICD10CM",
                "diagnosis_primary": int(code == principal),
                "poa_present": int((h or 0.0) < 24),
            })
        return rows

    def _generate_hospitalization_diagnoses(
        self,
        hospitalization_id: str,
    ) -> list[dict]:
        """Generate diagnoses for one hospitalization."""
        records = []
        has_principal = False

        # Shuffle diagnoses to randomize selection
        dx_items = list(self.DIAGNOSES.items())
        self.rng.shuffle(dx_items)

        for dx_name, dx_data in dx_items:
            if self.rng.random() < dx_data["probability"]:
                code = self.rng.choice(dx_data["codes"])

                # Determine if principal (CLIF 2.1.0: diagnosis_primary as int 0/1)
                if dx_data["is_principal"] and not has_principal:
                    dx_primary = 1
                    has_principal = True
                else:
                    dx_primary = 0

                # POA status (CLIF 2.1.0: poa_present as int 0/1)
                poa = int(self.rng.random() < 0.70)

                records.append(
                    {
                        "hospitalization_id": hospitalization_id,
                        "diagnosis_code": code,
                        "diagnosis_code_format": "ICD10CM",
                        "diagnosis_primary": dx_primary,
                        "poa_present": poa,
                    }
                )

        # Ensure at least one principal diagnosis
        if records and not has_principal:
            records[0]["diagnosis_primary"] = 1

        return records
