"""ADT (Admit/Discharge/Transfer) table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import course_for_row


class ADTGenerator(BaseGenerator):
    """Generate synthetic ADT events.

    Creates adt table with location transfers during hospitalization:
    - hospitalization_id (foreign key)
    - hospital_id, hospital_type (required for CLIF 2.1.0)
    - in_dttm, out_dttm (contiguous and within hospitalization bounds)
    - location_name, location_category, location_type (required)

    Typical flow: ED → ICU → Stepdown → Ward → Discharge
    """

    # Common location flow patterns using CLIF 2.1.0 permissible
    # location_category values (all lowercase)
    FLOW_PATTERNS = {
        "emergency_to_icu": ["ed", "icu", "stepdown", "ward"],
        "direct_icu": ["icu", "stepdown", "ward"],
        "stepdown_only": ["ed", "stepdown", "ward"],
        "ward_only": ["ed", "ward"],
        "short_icu": ["icu", "ward"],
    }

    FLOW_WEIGHTS = [0.45, 0.30, 0.05, 0.05, 0.15]

    # CLIF 2.1.0 hospital types
    HOSPITAL_TYPES = ["academic", "community", "LTACH"]
    HOSPITAL_TYPE_WEIGHTS = [0.6, 0.35, 0.05]

    # CLIF 2.1.0 ICU location types (for location_type column)
    LOCATION_TYPES = [
        "general_icu",
        "cardiac_icu",
        "cardiothoracic_surgical_icu",
        "mixed_cardiothoracic_icu",
        "surgical_icu",
        "burn_icu",
        "neuro_icu",
        "neurosurgical_icu",
        "mixed_neuro_icu",
        "medical_icu",
    ]
    LOCATION_TYPE_WEIGHTS = [0.25, 0.15, 0.05, 0.05, 0.15, 0.02, 0.08, 0.05, 0.05, 0.15]

    # Location names by category
    LOCATION_NAMES = {
        "ed": ["Emergency Department"],
        "stepdown": ["Stepdown Unit", "Progressive Care Unit"],
        "ward": ["Medical Ward", "Surgical Ward", "General Ward"],
        "procedural": ["Procedure Suite", "Operating Room"],
        "other": ["Other Unit"],
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate ADT events for each hospitalization.

        Args:
            hospitalizations_df: Hospitalization table DataFrame

        Returns:
            DataFrame with ADT table columns
        """
        records = []

        # Generate a set of hospital IDs (simulate multi-hospital system)
        n_hospitals = max(3, len(hospitalizations_df) // 100)
        hospital_ids = [f"HOSP-{i:03d}" for i in range(1, n_hospitals + 1)]
        hospital_types = self.rng.choice(
            self.HOSPITAL_TYPES,
            size=n_hospitals,
            p=self.HOSPITAL_TYPE_WEIGHTS,
        )
        hospital_map = dict(zip(hospital_ids, hospital_types))

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            # Handle missing discharge time
            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            # Ensure timezone
            if admit_time.tzinfo is None:
                admit_time = admit_time.replace(tzinfo=timezone.utc)
            if discharge_time.tzinfo is None:
                discharge_time = discharge_time.replace(tzinfo=timezone.utc)

            # Assign hospital for this hospitalization
            hospital_id = self.rng.choice(hospital_ids)
            hospital_type = hospital_map[hospital_id]

            # Locations follow the stay's clinical course: the patient is in
            # the ICU whenever the course needs ICU-level support.
            course = course_for_row(hosp)
            adt_events = self._generate_course_sequence(
                hosp_id, admit_time, discharge_time, hospital_id, hospital_type,
                course, hosp.get("admission_type_category"),
            )
            records.extend(adt_events)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["in_dttm"] = pd.to_datetime(df["in_dttm"], utc=True)
            df["out_dttm"] = pd.to_datetime(df["out_dttm"], utc=True)

        return df

    def _generate_course_sequence(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        hospital_id: str,
        hospital_type: str,
        course,
        admission_type=None,
    ) -> list[dict]:
        """ED or OR, then ICU while support is needed, then stepdown and ward."""
        los_h = (discharge_time - admit_time).total_seconds() / 3600
        if course is None or los_h <= 0:
            return self._generate_location_sequence(
                hospitalization_id, admit_time, discharge_time, hospital_id, hospital_type
            )
        n = course.n_hours
        need = (
            (course.level >= 2) | (course.ne_eq > 0) | course.crrt | course.prone
            | (course.severity > 0.6)
        )
        for m in course.mcs:
            need[m["start"]:m["end"]] = True

        segs = []  # (start_h, end_h, category)
        t0 = 0.0
        adm = str(admission_type or "").lower()
        if adm == "elective":
            t0 = self.rng.uniform(2, 6)
            segs.append((0.0, t0, "procedural"))
        elif adm in ("ed", "") or self.rng.random() < 0.15:
            t0 = self.rng.uniform(2, 8)
            segs.append((0.0, t0, "ed"))
        t0 = min(t0, los_h)

        icu = []
        runs = []
        start = None
        for i, v in enumerate(need):
            if v and start is None:
                start = i
            elif not v and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, n))
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] < 24:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for a, b in merged:
            lo = max(a - self.rng.uniform(0, 2), t0)
            hi = min(b + self.rng.uniform(12, 36), los_h)
            if hi > lo:
                icu.append([lo, hi])
        if not icu and (adm == "elective" or self.rng.random() < 0.7):
            # Observation-level ICU stay for patients who never needed support
            icu.append([t0, min(t0 + self.rng.uniform(12, 36), los_h)])
        if course.expired and icu and los_h - icu[-1][1] < 24:
            icu[-1][1] = los_h  # deaths happen in the ICU
        joined = []
        for lo, hi in icu:
            if joined and lo - joined[-1][1] < 12:
                joined[-1][1] = hi
            else:
                joined.append([lo, hi])

        cur = t0
        for k, (lo, hi) in enumerate(joined):
            lo = max(lo, cur)  # a stepdown stay can run into the next ICU stay
            if hi <= lo:
                continue
            if lo > cur:
                segs.append((cur, lo, "ward"))
            segs.append((lo, hi, "icu"))
            cur = hi
            if cur < los_h - 6 and self.rng.random() < 0.35:
                next_icu = joined[k + 1][0] if k + 1 < len(joined) else los_h
                sd = min(cur + self.rng.uniform(12, 48), los_h, next_icu)
                if sd > cur:
                    segs.append((cur, sd, "stepdown"))
                    cur = sd
        if cur < los_h:
            segs.append((cur, los_h, "ward" if joined or self.rng.random() < 0.6 else "stepdown"))

        icu_type = self._icu_type(course, adm)
        ward_name = str(self.rng.choice(["Medical Ward", "Surgical Ward", "General Ward"]))
        events = []
        for lo, hi, cat in segs:
            if hi - lo <= 0:
                continue
            if cat == "icu":
                loc_type = icu_type
                name = icu_type.replace("_", " ").title().replace("Icu", "ICU")
            else:
                loc_type = None
                name = {"ed": "Emergency Department", "stepdown": "Stepdown Unit",
                        "procedural": "Operating Room"}.get(cat, ward_name)
            if events and events[-1]["location_category"] == cat:
                events[-1]["out_dttm"] = admit_time + timedelta(hours=hi)
                continue
            events.append({
                "hospitalization_id": hospitalization_id,
                "hospital_id": hospital_id,
                "hospital_type": hospital_type,
                "in_dttm": admit_time + timedelta(hours=lo),
                "out_dttm": admit_time + timedelta(hours=hi),
                "location_name": name,
                "location_category": cat,
                "location_type": loc_type,
            })
        if events:
            events[0]["in_dttm"] = admit_time
            events[-1]["out_dttm"] = discharge_time
        return events

    def _icu_type(self, course, admission_type: str) -> str:
        if course.shock_type == "cardiogenic" and self.rng.random() < 0.7:
            return "cardiac_icu"
        if admission_type == "elective":
            return str(self.rng.choice(["surgical_icu", "cardiothoracic_surgical_icu", "neurosurgical_icu"],
                                       p=[0.5, 0.3, 0.2]))
        return str(self.rng.choice(["medical_icu", "general_icu", "mixed_neuro_icu", "surgical_icu"],
                                   p=[0.5, 0.3, 0.1, 0.1]))

    def _generate_location_sequence(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        hospital_id: str,
        hospital_type: str,
    ) -> list[dict]:
        """Generate sequence of location transfers."""
        total_hours = (discharge_time - admit_time).total_seconds() / 3600

        if total_hours <= 0:
            location_type = self.rng.choice(self.LOCATION_TYPES, p=self.LOCATION_TYPE_WEIGHTS)
            location_name = location_type.replace("_", " ").title().replace("Icu", "ICU")
            return [
                {
                    "hospitalization_id": hospitalization_id,
                    "hospital_id": hospital_id,
                    "hospital_type": hospital_type,
                    "in_dttm": admit_time,
                    "out_dttm": discharge_time,
                    "location_name": location_name,
                    "location_category": "icu",
                    "location_type": location_type,
                }
            ]

        # Select flow pattern based on weights
        pattern_names = list(self.FLOW_PATTERNS.keys())
        pattern_idx = self.rng.choice(len(pattern_names), p=self.FLOW_WEIGHTS)
        locations = self.FLOW_PATTERNS[pattern_names[pattern_idx]].copy()

        # Adjust pattern based on LOS
        if total_hours < 24:
            # Very short stay - single location
            locations = [locations[0] if locations else "icu"]
        elif total_hours < 72:
            # Short stay - max 2 locations
            locations = locations[:2] if len(locations) > 2 else locations

        n_locations = len(locations)

        # Distribute time across locations
        # ICU gets more time early, ward gets more time late
        time_weights = self._get_time_weights(locations, total_hours)
        location_hours = [w * total_hours for w in time_weights]

        # Generate ADT events
        events = []
        current_time = admit_time

        for i, (location, hours) in enumerate(zip(locations, location_hours)):
            end_time = current_time + timedelta(hours=hours)

            # Last location ends at discharge
            if i == n_locations - 1:
                end_time = discharge_time

            # Add some randomness to transfer times
            if i > 0 and i < n_locations - 1:
                jitter_hours = self.rng.uniform(-0.5, 0.5)
                end_time += timedelta(hours=jitter_hours)
                end_time = min(end_time, discharge_time)

            # Determine location_type and location_name
            if location == "icu":
                location_type = self.rng.choice(self.LOCATION_TYPES, p=self.LOCATION_TYPE_WEIGHTS)
                location_name = location_type.replace("_", " ").title().replace("Icu", "ICU")
            elif location == "ed":
                location_type = None
                location_name = "Emergency Department"
            elif location == "stepdown":
                location_type = None
                location_name = "Stepdown Unit"
            elif location == "ward":
                location_type = None
                location_name = self.rng.choice(["Medical Ward", "Surgical Ward", "General Ward"])
            else:
                location_type = None
                location_name = location.title()

            events.append(
                {
                    "hospitalization_id": hospitalization_id,
                    "hospital_id": hospital_id,
                    "hospital_type": hospital_type,
                    "in_dttm": current_time,
                    "out_dttm": end_time,
                    "location_name": location_name,
                    "location_category": location,
                    "location_type": location_type,
                }
            )

            current_time = end_time

        return events

    def _get_location_details(self, location: str) -> tuple[str, str]:
        """Get location_type and location_name for a location_category."""
        if location == "icu":
            loc_type = self.rng.choice(self.LOCATION_TYPES, p=self.LOCATION_TYPE_WEIGHTS)
            loc_name = loc_type.replace("_", " ").title().replace("Icu", "ICU")
            return loc_type, loc_name

        # For non-ICU locations, location_type is NULL per schema
        # (location_type only has ICU subtypes as permissible values)
        names = self.LOCATION_NAMES.get(location, ["Other Unit"])
        loc_name = self.rng.choice(names) if names else location.title()
        return None, loc_name

    def _get_time_weights(
        self, locations: list[str], total_hours: float
    ) -> list[float]:
        """Calculate time weights for each location."""
        n = len(locations)
        weights = []

        for i, loc in enumerate(locations):
            if loc == "ed":
                # ED: short stay (2-8 hours)
                weights.append(max(2, min(8, total_hours * 0.05)) / total_hours)
            elif loc == "icu":
                # ICU: variable, usually substantial portion
                weights.append(0.4 if n > 2 else 0.6)
            elif loc == "stepdown":
                weights.append(0.25)
            elif loc == "ward":
                weights.append(0.3)
            else:
                weights.append(1.0 / n)

        # Normalize
        total = sum(weights)
        return [w / total for w in weights]
