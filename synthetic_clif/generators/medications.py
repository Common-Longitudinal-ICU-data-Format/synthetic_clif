"""Medication administration generators (continuous and intermittent)."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.timestamps import generate_irregular_timestamps
from synthetic_clif.physiology.course import HospitalCourse, course_for_row


class MedicationContinuousGenerator(BaseGenerator):
    """Generate synthetic continuous medication administration data.

    Creates medication_admin_continuous table with:
    - Vasoactives that run while the shared course needs them to hold MAP,
      at the doses that produce the MAP seen in vitals
    - Sedation and analgesia during each IMV episode
    - Insulin infusions while glucose runs high
    - start / dose_change / going / stop MAR sequences
    """

    # Medication parameters: (typical_dose, unit, dose_range, titration_delta)
    MED_PARAMS = {
        "norepinephrine": {
            "dose_range": (0.01, 0.5),
            "unit": "mcg/kg/min",
            "titration": 0.02,
            "indication": "vasopressor",
        },
        "epinephrine": {
            "dose_range": (0.01, 0.3),
            "unit": "mcg/kg/min",
            "titration": 0.01,
            "indication": "vasopressor",
        },
        "vasopressin": {
            "dose_range": (0.01, 0.04),
            "unit": "units/min",
            "titration": 0.005,
            "indication": "vasopressor",
        },
        "dopamine": {
            "dose_range": (2, 20),
            "unit": "mcg/kg/min",
            "titration": 2,
            "indication": "vasopressor",
        },
        "dobutamine": {
            "dose_range": (2, 20),
            "unit": "mcg/kg/min",
            "titration": 2.5,
            "indication": "inotrope",
        },
        "phenylephrine": {
            "dose_range": (20, 200),
            "unit": "mcg/min",
            "titration": 20,
            "indication": "vasopressor",
        },
        "milrinone": {
            "dose_range": (0.125, 0.75),
            "unit": "mcg/kg/min",
            "titration": 0.125,
            "indication": "inotrope",
        },
        "propofol": {
            "dose_range": (5, 80),
            "unit": "mcg/kg/min",
            "titration": 10,
            "indication": "sedation",
        },
        "dexmedetomidine": {
            "dose_range": (0.2, 1.5),
            "unit": "mcg/kg/hr",
            "titration": 0.1,
            "indication": "sedation",
        },
        "midazolam": {
            "dose_range": (0.5, 10),
            "unit": "mg/hr",
            "titration": 1,
            "indication": "sedation",
        },
        "fentanyl": {
            "dose_range": (25, 200),
            "unit": "mcg/hr",
            "titration": 25,
            "indication": "analgesia",
        },
        "morphine": {
            "dose_range": (1, 10),
            "unit": "mg/hr",
            "titration": 1,
            "indication": "analgesia",
        },
        "heparin": {
            "dose_range": (500, 2000),
            "unit": "units/hr",
            "titration": 100,
            "indication": "anticoagulation",
        },
        "insulin": {
            "dose_range": (0.5, 10),
            "unit": "units/hr",
            "titration": 0.5,
            "indication": "glycemic",
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        respiratory_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Generate continuous medication administration.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            respiratory_df: Optional respiratory support for ventilation status

        Returns:
            DataFrame with medication_admin_continuous columns
        """
        records = []

        # respiratory_df is accepted for compatibility; IMV timing now comes
        # from the shared course, which the respiratory table also follows.

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            course = course_for_row(hosp)
            hosp_meds = self._generate_course_meds(hosp_id, course)
            # Heparin infusions are not driven by the course
            hosp_meds.extend(self._generate_heparin(hosp_id, admit_time, discharge_time))
            records.extend(hosp_meds)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["admin_dttm"] = pd.to_datetime(df["admin_dttm"], utc=True)

        return df

    def _safe_uniform(self, low: float, high: float) -> float:
        """Generate uniform random value, handling edge cases where high <= low."""
        if high <= low:
            return low
        return self.rng.uniform(low, high)

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        """Contiguous True runs as (start, end_exclusive) hour indices."""
        runs, start = [], None
        for i, v in enumerate(mask):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    def _order_id(self) -> str:
        return f"{int(self.rng.integers(0, 16**8)):08x}"

    def _chart_run(self, hosp_id, course: HospitalCourse, med: str, start_h: float,
                   end_h: float, dose_at) -> list[dict]:
        """MAR rows for one infusion: start, dose changes, going/verify, stop."""
        params = self.MED_PARAMS[med]
        order_id = self._order_id()
        t0 = course.admit_time
        # last charting just before discharge (n_hours runs past it)
        limit = min(course.n_hours - 2, course.los_h - 0.05) if course.los_h else course.n_hours - 2
        end_h = min(end_h, limit)
        if end_h - start_h < 0.5:
            return []
        hours = [start_h]
        h = start_h
        while True:
            h += self.rng.uniform(0.7, 1.3)
            if h >= end_h:
                break
            hours.append(h)
        rows, last = [], None
        for k, hh in enumerate(hours):
            dose = round(float(dose_at(hh)), 3)
            if k == 0:
                action = "start"
            elif last is not None and abs(dose - last) > 1e-9:
                action = "dose_change"
            elif self.rng.random() < 0.6:
                continue
            else:
                action = "verify" if self.rng.random() < 0.2 else "going"
            rows.append((hh, dose, action))
            last = dose
        rows.append((end_h, last, "stop"))
        out = []
        for hh, dose, action in rows:
            out.append({
                "hospitalization_id": hosp_id,
                "med_order_id": order_id,
                "admin_dttm": t0 + timedelta(hours=float(hh)),
                "med_category": med,
                "med_name": med.replace("_", "-").title(),
                "med_group": self._get_med_group_continuous(params["indication"]),
                "med_dose": dose,
                "med_dose_unit": params["unit"],
                "med_route_category": "iv",
                "med_route_name": self.name_from_category("iv"),
                "mar_action_category": action,
                "mar_action_name": self.name_from_category(action),
                "mar_action_group": "administered",
            })
        return out

    def _generate_course_meds(self, hosp_id, course: Optional[HospitalCourse]) -> list[dict]:
        """Vasoactives, sedation, analgesia and insulin that follow the course.

        Vasopressors run while the course needs them to hold MAP, at doses
        converted from the norepinephrine-equivalent dose that sets MAP in
        the vitals; sedation and analgesia run during each IMV episode.
        """
        if course is None or course.n_hours < 6:
            return []
        rec = []
        weight = course.weight_kg
        agent = course.pressor_agent
        ne = course.ne_eq

        def pressor_dose(hh):
            d = ne[min(int(hh), len(ne) - 1)]
            if agent in ("norepinephrine", "epinephrine"):
                return float(np.clip(round(d, 2), 0.01, 1.0))
            if agent == "phenylephrine":
                return float(np.clip(round(d * 10 * weight / 5) * 5, 20, 300))
            return float(np.clip(round(d * 100 * 2) / 2, 2, 20))  # dopamine

        for s, e in self._runs(ne > 0):
            rec += self._chart_run(hosp_id, course, agent, s + self.rng.uniform(0, 0.5), e, pressor_dose)
        vaso_dose = float(self.rng.choice([0.03, 0.04], p=[0.8, 0.2]))
        for s, e in self._runs(course.vasopressin):
            rec += self._chart_run(hosp_id, course, "vasopressin", s + self.rng.uniform(0, 0.5), e,
                                   lambda hh: vaso_dose)
        dobu_dose = float(self.rng.choice([2.5, 5.0, 7.5]))
        for s, e in self._runs(course.dobutamine):
            rec += self._chart_run(hosp_id, course, "dobutamine", s + self.rng.uniform(0.5, 2), e,
                                   lambda hh: dobu_dose)

        # Sedation and analgesia for each IMV episode, stopped around extubation.
        # Doses follow the course's sedation depth, which also sets RASS.
        for s, e in course.imv_episodes:
            if e - s < 4:
                continue
            for med in (course.sedative, course.analgesic):
                lo, hi = self.MED_PARAMS[med]["dose_range"]
                step = self.MED_PARAMS[med]["titration"]
                depth = course.sed_depth[s:min(e + 2, course.n_hours)]
                depth = np.concatenate([depth, np.repeat(depth[-1:], max(0, int(e - s) + 2 - len(depth)))])
                target = lo + (hi - lo) * (0.1 + 0.7 * depth)
                walk = np.clip(np.round(target / step) * step, lo, hi)
                if med == "propofol" and (ne[s:e] > 0).any():  # lighter propofol on vasopressors
                    walk = np.maximum(walk * 0.7, lo)
                start = s + self.rng.uniform(0, 1)
                end = e - self.rng.uniform(0.5, 2) if e < course.n_hours - 2 else e
                rec += self._chart_run(hosp_id, course, med, start, end,
                                       lambda hh, w=walk, s0=s: max(w[min(int(hh - s0), len(w) - 1)], lo))

        # Insulin when glucose runs high
        for s, e in self._runs(course.insulin):
            if e - s < 3:
                continue
            base = self.rng.uniform(1, 4)
            rec += self._chart_run(hosp_id, course, "insulin", s + self.rng.uniform(0, 1), e,
                                   lambda hh, b0=base: float(np.clip(round((b0 + 2 * course.severity[min(int(hh), course.n_hours - 1)]) * 2) / 2, 0.5, 10)))
        return rec

    def _generate_heparin(self, hosp_id, admit_time, discharge_time) -> list[dict]:
        los_hours = (discharge_time - admit_time).total_seconds() / 3600
        if los_hours < 24 or self.rng.random() >= 0.15:
            return []
        start_offset = self._safe_uniform(0, min(48, los_hours / 2))
        return self._generate_infusion(
            hosp_id,
            admit_time + timedelta(hours=start_offset),
            discharge_time,
            "heparin",
            duration_hours=self._safe_uniform(24, min(120, los_hours - start_offset)),
        )

    def _generate_infusion(
        self,
        hospitalization_id: str,
        start_time: datetime,
        end_time: datetime,
        medication: str,
        duration_hours: float,
    ) -> list[dict]:
        """Generate infusion records with titration."""
        records = []
        params = self.MED_PARAMS.get(medication)
        if params is None:
            return records

        order_id = self._order_id()
        actual_end = min(start_time + timedelta(hours=duration_hours), end_time)

        # Generate dose changes over time (titration pattern)
        timestamps = generate_irregular_timestamps(
            start_time,
            actual_end,
            mean_interval_hours=1.0,
            cv=0.4,
            rng=self.rng,
        )

        dose_range = params["dose_range"]
        titration = params["titration"]
        current_dose = self.rng.uniform(dose_range[0], (dose_range[0] + dose_range[1]) / 2)

        for ts in timestamps:
            # Titration: small dose adjustments
            if self.rng.random() < 0.3:
                delta = self.rng.choice([-1, 0, 1]) * titration
                current_dose = np.clip(
                    current_dose + delta, dose_range[0], dose_range[1]
                )

            mar_action_cat = self.rng.choice(
                ["dose_change", "going", "start", "stop", "verify"],
                p=[0.3, 0.4, 0.1, 0.1, 0.1],
            )
            med_route_cat = "iv"

            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "med_order_id": order_id,
                    "admin_dttm": ts,
                    "med_category": medication,
                    "med_name": medication.replace("_", "-").title(),
                    "med_group": self._get_med_group_continuous(params["indication"]),
                    "med_dose": round(current_dose, 3),
                    "med_dose_unit": params["unit"],
                    "med_route_category": med_route_cat,
                    "med_route_name": self.name_from_category(med_route_cat),
                    "mar_action_category": mar_action_cat,
                    "mar_action_name": self.name_from_category(mar_action_cat),
                    "mar_action_group": "administered",
                }
            )

        return records

    @staticmethod
    def _get_med_group_continuous(indication: str) -> str:
        """Map indication to CLIF 2.1.0 med_group."""
        mapping = {
            "vasopressor": "vasoactives",
            "inotrope": "vasoactives",
            "sedation": "sedation",
            "analgesia": "sedation",
            "anticoagulation": "anticoagulation",
            "glycemic": "endocrine",
        }
        return mapping.get(indication, "others")


class MedicationIntermittentGenerator(BaseGenerator):
    """Generate synthetic intermittent medication administration data.

    Creates medication_admin_intermittent table with:
    - Scheduled medications (antibiotics, PPIs)
    - PRN medications
    - MAR action categories (given, held, refused)
    """

    # Common intermittent medications
    MED_SCHEDULES = {
        "vancomycin": {
            "dose_range": (1000, 1500),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 12,
            "indication": "antibiotic",
        },
        "piperacillin_tazobactam": {
            "dose_range": (3375, 4500),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 6,
            "indication": "antibiotic",
        },
        "cefepime": {
            "dose_range": (1000, 2000),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 8,
            "indication": "antibiotic",
        },
        "meropenem": {
            "dose_range": (1000, 2000),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 8,
            "indication": "antibiotic",
        },
        "acetaminophen": {
            "dose_range": (500, 1000),
            "unit": "mg",
            "route": "enteral",
            "frequency_hours": 6,
            "indication": "analgesic",
        },
        "hydromorphone": {
            "dose_range": (1, 4),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 4,
            "indication": "analgesic",
        },
        "ketamine": {
            "dose_range": (10, 30),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 8,
            "indication": "analgesic",
        },
        "diphenhydramine": {
            "dose_range": (25, 50),
            "unit": "mg",
            "route": "IV",
            "frequency_hours": 8,
            "indication": "analgesic",
        },
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate intermittent medication administration.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with medication_admin_intermittent columns
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
            if course is None:
                hosp_meds = self._generate_hospitalization_meds(hosp_id, admit_time, discharge_time)
            else:
                hosp_meds = self._course_meds(
                    hosp_id, admit_time, discharge_time, course,
                    str(hosp.get("admission_type_category") or ""),
                )
            records.extend(hosp_meds)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["admin_dttm"] = pd.to_datetime(df["admin_dttm"], utc=True)

        return df

    # Usual interval (h) with normal and reduced kidney function
    ABX_INTERVAL = {
        "vancomycin": (12, 24),
        "piperacillin_tazobactam": (6, 8),
        "cefepime": (8, 12),
        "meropenem": (8, 12),
    }

    def _dose_row(self, hosp_id, order_id, ts, med, dose, route, action=None) -> dict:
        params = self.MED_SCHEDULES[med]
        if action is None:
            action = str(self.rng.choice(["given", "not_given", "other"], p=[0.95, 0.04, 0.01]))
        mar_group = "administered" if action in ("given", "bolus") else (
            "not_administered" if action == "not_given" else "other")
        return {
            "hospitalization_id": hosp_id,
            "med_order_id": order_id,
            "admin_dttm": ts,
            "med_category": med,
            "med_name": med.replace("_", "-").title(),
            "med_group": self._get_med_group_intermittent(params["indication"]),
            "med_dose": float(dose),
            "med_dose_unit": params["unit"],
            "med_route_category": route,
            "med_route_name": self.name_from_category(route),
            "mar_action_category": action,
            "mar_action_name": self.name_from_category(action),
            "mar_action_group": mar_group,
        }

    def _order_id(self) -> str:
        return f"{int(self.rng.integers(0, 16**8)):08x}"

    def _course_meds(self, hosp_id, admit_time, discharge_time, course, admission_type) -> list[dict]:
        """Intermittent doses that follow the course.

        Antibiotics run from infection recognition for the course's duration,
        dosed by weight and spaced out when the kidneys fail; acetaminophen is
        given for fever; opioids for pain in patients who are awake; ketamine
        at intubation.
        """
        rows = []
        n = course.n_hours
        at = lambda h: admit_time + timedelta(hours=float(h))
        npo = (course.level >= 3) | (course.severity > 0.7)

        for abx in course.antibiotics:
            med = abx["agent"]
            oid = self._order_id()
            normal, reduced = self.ABX_INTERVAL[med]
            if med == "vancomycin":
                dose = float(np.clip(round(17.5 * course.weight_kg / 250) * 250, 750, 2500))
            else:
                dose = {"piperacillin_tazobactam": 4500, "cefepime": 2000, "meropenem": 1000}[med]
            h = abx["start"]
            while h < abx["end"] and at(h) < discharge_time:
                if med == "vancomycin" and h == abx["start"]:
                    give = float(min(round(25 * course.weight_kg / 250) * 250, 3000))  # loading dose
                else:
                    give = dose
                rows.append(self._dose_row(hosp_id, oid, at(h), med, give, "iv"))
                i = min(int(h), n - 1)
                kidney_low = course.aki[i] > 0.5 and course.crrt_effect(i) < 0.5
                h += (reduced if kidney_low else normal) + self.rng.uniform(-0.5, 0.5)

        # Acetaminophen for fever (at most every 6 h), scheduled after surgery
        apap_oid, last = self._order_id(), -99.0
        for i in range(n):
            febrile = course.temp[i] > 38.3
            post_op = admission_type == "elective" and i < 72 and i % 6 == 0
            if (febrile or post_op) and i - last >= 6 and at(i) < discharge_time:
                route = "iv" if npo[i] and course.level[i] != 4 else "enteral"
                rows.append(self._dose_row(hosp_id, apap_oid, at(i + self.rng.uniform(0, 1)),
                                           "acetaminophen", 650 if course.weight_kg < 60 else 1000, route))
                last = i

        # Hydromorphone for pain in patients who can ask for it
        hm_oid = self._order_id()
        pain_prone = 0.08 + 0.25 * (admission_type == "elective")
        for i in range(0, n, 4):
            if course.level[i] < 4 and course.rass[i] >= -1 and self.rng.random() < pain_prone:
                if at(i) < discharge_time:
                    rows.append(self._dose_row(hosp_id, hm_oid, at(i + self.rng.uniform(0, 3)),
                                               "hydromorphone", float(self.rng.choice([0.5, 1.0, 2.0])), "iv"))

        # Ketamine as an induction agent at intubation
        for s, _ in course.imv_episodes:
            if s > 0 and self.rng.random() < 0.4:
                rows.append(self._dose_row(hosp_id, self._order_id(), at(s - self.rng.uniform(0, 0.2)),
                                           "ketamine", float(round(1.5 * course.weight_kg / 10) * 10), "iv",
                                           action="given"))

        # Diphenhydramine as needed for a few patients (itch, sleep)
        if self.rng.random() < 0.1:
            dph_oid = self._order_id()
            for _ in range(int(self.rng.integers(1, 4))):
                h = self.rng.uniform(0, n - 1)
                if at(h) < discharge_time:
                    rows.append(self._dose_row(hosp_id, dph_oid, at(h), "diphenhydramine", 25.0, "iv"))
        # jittered dose times can land just past discharge
        return [r for r in rows if r["admin_dttm"] < discharge_time]

    def _generate_hospitalization_meds(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
    ) -> list[dict]:
        """Generate intermittent meds for one hospitalization."""
        records = []

        # Scheduled acetaminophen for most patients
        if self.rng.random() < 0.85:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "acetaminophen",
                )
            )

        # Antibiotics: ~50% of patients
        if self.rng.random() < 0.5:
            # Choose antibiotic regimen
            if self.rng.random() < 0.4:
                # Vancomycin + piperacillin-tazobactam (common broad spectrum)
                records.extend(
                    self._generate_scheduled_med(
                        hospitalization_id,
                        admit_time,
                        discharge_time,
                        "vancomycin",
                        duration_days=self.rng.uniform(5, 14),
                    )
                )
                records.extend(
                    self._generate_scheduled_med(
                        hospitalization_id,
                        admit_time,
                        discharge_time,
                        "piperacillin_tazobactam",
                        duration_days=self.rng.uniform(5, 14),
                    )
                )
            else:
                # Single agent
                abx = self.rng.choice(["cefepime", "meropenem"])
                records.extend(
                    self._generate_scheduled_med(
                        hospitalization_id,
                        admit_time,
                        discharge_time,
                        abx,
                        duration_days=self.rng.uniform(5, 10),
                    )
                )

        # Antihistamine (common scheduled med)
        if self.rng.random() < 0.7:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "diphenhydramine",
                )
            )

        # PRN analgesics for some patients
        if self.rng.random() < 0.3:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "hydromorphone",
                )
            )
        if self.rng.random() < 0.2:
            records.extend(
                self._generate_scheduled_med(
                    hospitalization_id,
                    admit_time,
                    discharge_time,
                    "ketamine",
                )
            )

        return records

    def _generate_scheduled_med(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        medication: str,
        duration_days: Optional[float] = None,
    ) -> list[dict]:
        """Generate scheduled medication doses."""
        records = []
        params = self.MED_SCHEDULES.get(medication)
        if params is None:
            return records

        order_id = f"{int(self.rng.integers(0, 16**8)):08x}"
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        if duration_days:
            end_time = min(
                admit_time + timedelta(days=duration_days), discharge_time
            )
        else:
            end_time = discharge_time

        freq_hours = params["frequency_hours"]
        dose_range = params["dose_range"]
        dose = self.rng.uniform(dose_range[0], dose_range[1])

        current_time = admit_time
        while current_time < end_time:
            # Determine MAR action
            action = self._sample_mar_action()

            mar_group = "administered" if action in ["given", "bolus"] else (
                "not_administered" if action == "not_given" else "other"
            )
            med_route_cat = self._map_route(params["route"].lower())
            records.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "med_order_id": order_id,
                    "admin_dttm": current_time,
                    "med_category": medication,
                    "med_name": medication.replace("_", "-").title(),
                    "med_group": self._get_med_group_intermittent(params["indication"]),
                    "med_dose": round(dose, 0),
                    "med_dose_unit": params["unit"],
                    "med_route_category": med_route_cat,
                    "med_route_name": self.name_from_category(med_route_cat),
                    "mar_action_category": action,
                    "mar_action_name": self.name_from_category(action),
                    "mar_action_group": mar_group,
                }
            )

            # Add jitter to scheduled time
            jitter = self.rng.uniform(-0.5, 0.5)
            current_time += timedelta(hours=freq_hours + jitter)

        return records

    def _sample_mar_action(self) -> str:
        """Sample MAR action per CLIF 2.1.0 schema."""
        return self.rng.choice(
            ["given", "bolus", "not_given", "other"],
            p=[0.88, 0.04, 0.06, 0.02],
        )

    @staticmethod
    def _map_route(route: str) -> str:
        """Map route abbreviations to CLIF 2.1.0 permissible med_route_category values.

        Permissible values: enteral, im, iv, buccal_sublingual, intrapleural.
        """
        route_mapping = {
            "iv": "iv",
            "im": "im",
            "po": "enteral",
            "sc": "im",
            "enteral": "enteral",
            "buccal_sublingual": "buccal_sublingual",
            "intrapleural": "intrapleural",
        }
        return route_mapping.get(route, "iv")

    @staticmethod
    def _get_med_group_intermittent(indication: str) -> str:
        """Map indication to CLIF 2.1.0 med_group for intermittent meds."""
        mapping = {
            "antibiotic": "CMS_sepsis_qualifying_antibiotics",
            "ppi": "other",
            "analgesic": "other",
            "prophylaxis": "other",
        }
        return mapping.get(indication, "other")
