"""Vitals table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import HospitalCourse, course_for_row
from synthetic_clif.utils.timestamps import generate_irregular_timestamps


class VitalsGenerator(BaseGenerator):
    """Generate synthetic vital signs.

    Creates vitals table with:
    - hospitalization_id (foreign key)
    - recorded_dttm (irregular timestamps, ~hourly in ICU, ~q4h elsewhere)
    - vital_category (mCIDE: temp_c, heart_rate, sbp, dbp, spo2, etc.)
    - vital_value (realistic ranges with temporal autocorrelation)
    - meas_site_name (measurement site)

    Features:
    - Values observed from the hospitalization's shared course
      (synthetic_clif.physiology), so vitals agree with vasopressors,
      FiO2 and ventilator settings
    - MAP derived from the same SBP/DBP reading (DBP + pulse pressure / 3)
    - Measurement frequency varies by location and instability
    - ~5% missingness, ~1% artefacts (BP triplets dropped or distorted together)
    """

    # Vital sign parameters: (mean, std, lower_bound, upper_bound)
    VITAL_PARAMS = {
        "temp_c": (37.0, 0.5, 34.0, 42.0),
        "heart_rate": (80.0, 15.0, 30.0, 200.0),
        "sbp": (120.0, 20.0, 60.0, 250.0),
        "dbp": (75.0, 12.0, 30.0, 150.0),
        "spo2": (96.0, 3.0, 50.0, 100.0),
        "respiratory_rate": (16.0, 4.0, 6.0, 50.0),
        "map": (90.0, 15.0, 40.0, 160.0),
        "height_cm": (170.0, 10.0, 140.0, 210.0),
        "weight_kg": (80.0, 20.0, 40.0, 200.0),
    }

    # Measurement frequency in hours by location_category
    FREQUENCY_BY_LOCATION = {
        "icu": 1.0,
        "ed": 1.0,
        "stepdown": 2.0,
        "ward": 4.0,
        "other": 4.0,
    }

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        adt_df: Optional[pd.DataFrame] = None,
        missingness_rate: float = 0.05,
        outlier_rate: float = 0.01,
    ) -> pd.DataFrame:
        """Generate vital signs for hospitalizations.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            adt_df: Optional ADT table for location-based frequency
            missingness_rate: Proportion of missing values
            outlier_rate: Proportion of outlier values

        Returns:
            DataFrame with vitals table columns
        """
        records = []

        # Build location timeline lookup
        location_lookup = self._build_location_lookup(adt_df)

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            # Generate vitals for this hospitalization from its shared course
            course = course_for_row(hosp)
            hosp_vitals = self._generate_hospitalization_vitals(
                hosp_id, admit_time, discharge_time, location_lookup.get(hosp_id), course
            )
            records.extend(hosp_vitals)

        df = pd.DataFrame(records)

        if len(df) > 0:
            # Add vital_name as lowercase version of vital_category
            df["vital_name"] = df["vital_category"].str.lower()
            df["recorded_dttm"] = pd.to_datetime(df["recorded_dttm"], utc=True)

            # Missingness and outliers. The SBP/DBP/MAP triplet from one cuff or
            # arterial-line reading is treated as a unit, so a dropped or
            # artefactual reading stays internally consistent.
            is_bp = df["_bp"] >= 0
            other = df.loc[~is_bp].copy()
            other = self.add_missingness(other, "vital_value", missingness_rate)
            outlier_bounds = {
                "temp_c": (32.0, 44.0),
                "heart_rate": (0.0, 300.0),
                "spo2": (50.0, 100.0),
                "respiratory_rate": (0.0, 60.0),
                "height_cm": (76.0, 255.0),
                "weight_kg": (30.0, 1100.0),
            }
            for cat, (lb, ub) in outlier_bounds.items():
                mask = other["vital_category"] == cat
                if mask.any():
                    subset = other.loc[mask].copy()
                    subset = self.add_outliers(
                        subset, "vital_value", outlier_rate,
                        lower_bound=lb, upper_bound=ub,
                    )
                    other.loc[mask, "vital_value"] = subset["vital_value"]
            df.loc[~is_bp, "vital_value"] = other["vital_value"]
            df = self._bp_missingness_and_artefacts(df, missingness_rate, outlier_rate)
            df = df.drop(columns="_bp")

        return df

    def _build_location_lookup(
        self, adt_df: Optional[pd.DataFrame]
    ) -> dict[str, list[tuple]]:
        """Build lookup for hospitalization locations over time."""
        if adt_df is None or len(adt_df) == 0:
            return {}

        lookup = {}
        for hosp_id in adt_df["hospitalization_id"].unique():
            hosp_adt = adt_df[adt_df["hospitalization_id"] == hosp_id]
            locations = []
            for _, row in hosp_adt.iterrows():
                locations.append(
                    (row["in_dttm"], row["out_dttm"], row["location_category"])
                )
            lookup[hosp_id] = sorted(locations, key=lambda x: x[0])

        return lookup

    def _get_location_at_time(
        self, time: datetime, locations: Optional[list[tuple]]
    ) -> str:
        """Get location at a specific time."""
        if locations is None:
            return "icu"

        for in_time, out_time, location in locations:
            if in_time <= time <= out_time:
                return location

        return "icu"

    def _bp_missingness_and_artefacts(
        self, df: pd.DataFrame, missingness_rate: float, outlier_rate: float
    ) -> pd.DataFrame:
        """Drop or distort whole SBP/DBP/MAP triplets."""
        groups = df.loc[df["_bp"] >= 0, "_bp"].unique()
        if len(groups) == 0:
            return df
        u = self.rng.random(len(groups))
        missing = set(groups[u < missingness_rate])
        artefact = groups[(u >= missingness_rate) & (u < missingness_rate + outlier_rate)]
        if missing:
            df.loc[df["_bp"].isin(missing), "vital_value"] = np.nan
        for g in artefact:
            rows = df.index[df["_bp"] == g]
            vals = dict(zip(df.loc[rows, "vital_category"], df.loc[rows, "vital_value"]))
            if "sbp" not in vals or "dbp" not in vals:
                continue
            if self.rng.random() < 0.5:   # line flush / transducer artefact: very high
                sbp = self.rng.uniform(220, 300)
                dbp = vals["dbp"] * self.rng.uniform(1.3, 2.0)
            else:                         # damped trace or cuff error: very low
                sbp = vals["sbp"] * self.rng.uniform(0.35, 0.6)
                dbp = vals["dbp"] * self.rng.uniform(0.5, 0.8)
            dbp = min(dbp, sbp - 5)
            new = {"sbp": round(sbp), "dbp": round(max(dbp, 5)),
                   "map": round(max(dbp, 5) + (sbp - max(dbp, 5)) / 3)}
            for r, cat in zip(rows, df.loc[rows, "vital_category"]):
                df.at[r, "vital_value"] = new[cat]
        return df

    def _generate_hospitalization_vitals(
        self,
        hospitalization_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        locations: Optional[list[tuple]],
        course: Optional[HospitalCourse] = None,
    ) -> list[dict]:
        """Generate all vital signs for one hospitalization by observing its course."""
        records = []
        if course is None:
            course = course_for_row({
                "hospitalization_id": hospitalization_id,
                "admission_dttm": admit_time,
                "discharge_dttm": discharge_time,
            })

        # Arterial line placed when the patient first needs pressors or IMV
        unstable = (course.ne_eq > 0) | (course.level == 4)
        art_from = None
        if unstable.any() and self.rng.random() < 0.85:
            art_from = int(np.argmax(unstable))

        timestamps = generate_irregular_timestamps(
            admit_time,
            discharge_time,
            mean_interval_hours=1.0,
            cv=0.3,
            rng=self.rng,
        )
        first = True
        for ts in timestamps:
            h = course.hour_index(ts)
            location = self._get_location_at_time(ts, locations)
            freq = self.FREQUENCY_BY_LOCATION.get(location, 4.0)
            # Sicker patients are watched more closely wherever they are
            if not (unstable[h] or course.level[h] >= 3 or first):
                if self.rng.random() > 1.0 / freq:
                    continue

            if self.rng.random() < 0.95:
                arterial = art_from is not None and h >= art_from
                records.extend(
                    self._generate_core_vitals(hospitalization_id, ts, course, arterial)
                )

            if self.rng.random() < 0.3 or first:
                records.append(
                    self._row(hospitalization_id, ts, "temp_c",
                              round(course.at("temp", ts) + self.rng.normal(0, 0.1), 1),
                              self.rng.choice(["Oral", "Tympanic", "Temporal", "Axillary"]))
                )

            if first:
                records.extend(self._generate_height_weight(hospitalization_id, ts, course))
            first = False

        return records

    @staticmethod
    def _row(hosp_id, ts, cat, value, site=None, bp_group=-1) -> dict:
        return {
            "hospitalization_id": hosp_id,
            "recorded_dttm": ts,
            "vital_category": cat,
            "vital_value": value,
            "meas_site_name": site,
            "_bp": bp_group,
        }

    def _generate_core_vitals(
        self,
        hospitalization_id: str,
        timestamp: datetime,
        course: HospitalCourse,
        arterial: bool = False,
    ) -> list[dict]:
        """Observe heart rate, blood pressure, SpO2 and respiratory rate."""
        hr = course.at("hr", timestamp) + self.rng.normal(0, 2)
        sbp = course.at("sbp", timestamp) + self.rng.normal(0, 2 if arterial else 4)
        dbp = course.at("dbp", timestamp) + self.rng.normal(0, 1.5 if arterial else 3)
        dbp = min(dbp, sbp - 10)
        # MAP is the time-average of the pressure wave: about DBP + pulse pressure / 3
        mean_ap = dbp + (sbp - dbp) / 3 + self.rng.normal(0, 1.5)
        mean_ap = float(np.clip(mean_ap, dbp + 1, sbp - 1))
        site = "Arterial" if arterial else None
        self._bp_counter = getattr(self, "_bp_counter", 0) + 1
        g = self._bp_counter

        vitals = [
            self._row(hospitalization_id, timestamp, "heart_rate", round(hr, 0)),
            self._row(hospitalization_id, timestamp, "sbp", round(sbp, 0), site, g),
            self._row(hospitalization_id, timestamp, "dbp", round(dbp, 0), site, g),
        ]
        if arterial or self.rng.random() < 0.9:
            vitals.append(self._row(hospitalization_id, timestamp, "map", round(mean_ap, 0), site, g))
        spo2 = float(np.clip(course.at("spo2", timestamp) + self.rng.normal(0, 1.0), 50, 100))
        vitals.append(self._row(hospitalization_id, timestamp, "spo2", round(spo2, 0)))
        rr = course.at("rr", timestamp) + self.rng.normal(0, 1.0)
        vitals.append(self._row(hospitalization_id, timestamp, "respiratory_rate", round(max(rr, 4), 0)))
        return vitals

    def _generate_height_weight(
        self,
        hospitalization_id: str,
        timestamp: datetime,
        course: HospitalCourse,
    ) -> list[dict]:
        """Admission height and weight (the same body the ventilator settings assume)."""
        return [
            self._row(hospitalization_id, timestamp, "height_cm", course.height_cm),
            self._row(hospitalization_id, timestamp, "weight_kg", course.weight_kg),
        ]
