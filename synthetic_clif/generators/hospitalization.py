"""Hospitalization table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.utils.distributions import log_normal_los

# CLIF research projects typically filter for 2018–2024 historical data
CLIF_DATE_START = datetime(2018, 1, 1, tzinfo=timezone.utc)
CLIF_DATE_END = datetime(2024, 12, 31, tzinfo=timezone.utc)


class HospitalizationGenerator(BaseGenerator):
    """Generate synthetic hospitalizations.

    Creates hospitalization table with:
    - hospitalization_id (includes patient prefix for traceability)
    - patient_id (foreign key to patient table)
    - admission_dttm, discharge_dttm (LOS follows log-normal distribution)
    - age_at_admission (computed from patient birth_date)
    - admission_type_category, discharge_category (mCIDE categories)
    """

    def generate(
        self,
        patients_df: pd.DataFrame,
        n_hospitalizations: int,
        reference_date: Optional[datetime] = None,
        median_los_days: float = 6.4,
        los_sigma: float = 1.4,
    ) -> pd.DataFrame:
        """Generate hospitalizations linked to patients.

        Args:
            patients_df: Patient table DataFrame
            n_hospitalizations: Total number of hospitalizations to generate
            reference_date: Unused; admissions are drawn uniformly from
                CLIF_DATE_START–CLIF_DATE_END (2018-01-01 to 2024-12-31)
                so that synthetic data matches the date range expected by
                CLIF research projects.
            median_los_days: Median length of stay in days
            los_sigma: Log-normal sigma parameter for LOS distribution

        Returns:
            DataFrame with hospitalization table columns
        """

        patient_ids = patients_df["patient_id"].tolist()
        birth_dates = patients_df["birth_date"].tolist()
        death_dttms = patients_df["death_dttm"].tolist()

        n_patients = len(patient_ids)

        # Distribute hospitalizations across patients
        # Some patients have multiple admissions, most have 1
        hosp_counts = self._distribute_hospitalizations(n_patients, n_hospitalizations)

        records = []
        for pt_idx, n_hosp in enumerate(hosp_counts):
            if n_hosp == 0:
                continue

            patient_id = patient_ids[pt_idx]
            birth_date = birth_dates[pt_idx]
            death_dttm = death_dttms[pt_idx]

            # Generate admission times uniformly within the CLIF historical range
            admission_times = self._generate_admission_times(n_hosp)

            # Generate LOS for each hospitalization
            los_days = log_normal_los(
                n_hosp,
                median_days=median_los_days,
                sigma=los_sigma,
                rng=self.rng,
            )

            prev_discharge = None
            for hosp_idx, (admit_time, los) in enumerate(
                zip(admission_times, los_days)
            ):
                # A readmission cannot start before the previous stay ends:
                # move it to 1-14 days after that discharge (no new random
                # draw, so the rest of the dataset is unchanged)
                if prev_discharge is not None and admit_time <= prev_discharge:
                    gap_days = 1 + int(float(los) * 1000) % 14
                    admit_time = (prev_discharge + timedelta(days=gap_days)).replace(
                        second=0, microsecond=0
                    )
                # Calculate discharge time
                discharge_time = admit_time + timedelta(days=float(los))

                # Determine discharge category
                is_terminal = (
                    death_dttm is not None
                    and pd.notna(death_dttm)
                    and admit_time <= death_dttm <= discharge_time
                )

                if is_terminal:
                    discharge_category = "Expired"
                    discharge_name = "expired"
                    discharge_time = death_dttm
                else:
                    discharge_category = self._sample_discharge_category()
                    discharge_name = self._get_discharge_name(discharge_category)

                # Calculate age at admission
                if pd.notna(birth_date):
                    age_at_admission = int(
                        (admit_time.date() - birth_date.date()).days / 365.25
                    )
                else:
                    age_at_admission = None

                # Generate hospitalization ID (includes patient prefix)
                hosp_id = f"{patient_id[:8]}-H{hosp_idx + 1:03d}"

                admission_type_cat = self._sample_admission_type()

                # Generate synthetic geographic data
                state_fips = f"{self.rng.integers(1, 56):02d}"
                county_fips = f"{self.rng.integers(1, 999):03d}"
                tract = f"{self.rng.integers(100000, 999999):06d}"
                zip5 = f"{self.rng.integers(10000, 99999)}"

                prev_discharge = discharge_time
                records.append(
                    {
                        "hospitalization_id": hosp_id,
                        "hospitalization_joined_id": hosp_id,
                        "patient_id": patient_id,
                        "admission_dttm": admit_time,
                        "discharge_dttm": discharge_time,
                        "age_at_admission": age_at_admission,
                        "admission_type_category": admission_type_cat,
                        "admission_type_name": self.name_from_category(admission_type_cat),
                        "discharge_name": discharge_name,
                        "discharge_category": discharge_category,
                        "zipcode_five_digit": zip5,
                        "zipcode_nine_digit": f"{zip5}{self.rng.integers(1000, 9999)}",
                        "census_block_code": f"{state_fips}{county_fips}{tract}{self.rng.integers(1000, 9999)}",
                        "census_block_group_code": f"{state_fips}{county_fips}{tract[:5]}{self.rng.integers(1, 9)}",
                        "census_tract": f"{state_fips}{county_fips}{tract}",
                        "state_code": state_fips,
                        "county_code": f"{state_fips}{county_fips}",
                        "fips_version": "2020",
                    }
                )

        df = pd.DataFrame(records)

        # Ensure datetime columns are UTC
        df["admission_dttm"] = pd.to_datetime(df["admission_dttm"], utc=True)
        df["discharge_dttm"] = pd.to_datetime(df["discharge_dttm"], utc=True)

        # Add missingness
        df = self.add_missingness(df, "age_at_admission", 0.01)
        df = self.add_missingness(df, "admission_type_category", 0.02)
        df = self.add_missingness(df, "discharge_name", 0.02)

        # Cast age_at_admission to nullable Int64 to preserve integer type with NaN
        if "age_at_admission" in df.columns:
            df["age_at_admission"] = df["age_at_admission"].astype("Int64")

        return df

    def _distribute_hospitalizations(
        self, n_patients: int, n_hospitalizations: int
    ) -> list[int]:
        """Distribute hospitalizations across patients.

        Most patients have 1 admission, some have multiple (readmissions).
        Uses a geometric-like distribution.
        """
        if n_hospitalizations <= n_patients:
            # Each selected patient gets exactly 1
            counts = [0] * n_patients
            selected = self.rng.choice(
                n_patients, size=n_hospitalizations, replace=False
            )
            for idx in selected:
                counts[idx] = 1
            return counts

        # Start with everyone getting 1
        counts = [1] * n_patients
        remaining = n_hospitalizations - n_patients

        # Distribute remaining hospitalizations (readmissions)
        # Probability of readmission decreases geometrically
        while remaining > 0:
            # 30% of patients have readmissions
            # max(1, ...) so fewer than 4 patients cannot loop forever
            n_readmit = min(remaining, max(1, int(n_patients * 0.3)))
            readmit_indices = self.rng.choice(n_patients, size=n_readmit, replace=True)
            for idx in readmit_indices:
                counts[idx] += 1
                remaining -= 1
                if remaining == 0:
                    break

        return counts

    def _generate_admission_times(self, n: int) -> list[datetime]:
        """Generate admission times drawn uniformly from CLIF_DATE_START to CLIF_DATE_END."""
        total_seconds = int(
            (CLIF_DATE_END - CLIF_DATE_START).total_seconds()
        )
        admission_times = []
        for _ in range(n):
            offset_seconds = int(self.rng.integers(0, total_seconds))
            admit_time = CLIF_DATE_START + timedelta(seconds=offset_seconds)
            # Round to minute boundary for realistic timestamps
            admit_time = admit_time.replace(second=0, microsecond=0)
            admission_times.append(admit_time)

        # Sort chronologically for same patient
        return sorted(admission_times)

    # CLIF 2.1.0 schema permissible values
    ADMISSION_TYPE_CATEGORIES = ["ed", "facility", "osh", "direct", "elective", "other"]
    DISCHARGE_CATEGORIES = [
        "Home",
        "Skilled Nursing Facility (SNF)",
        "Expired",
        "Acute Inpatient Rehab Facility",
        "Hospice",
        "Long Term Care Hospital (LTACH)",
        "Acute Care Hospital",
        "Group Home",
        "Chemical Dependency",
        "Against Medical Advice (AMA)",
        "Assisted Living",
        "Still Admitted",
        "Missing",
        "Other",
        "Psychiatric Hospital",
        "Shelter",
        "Jail",
    ]

    def _sample_admission_type(self) -> str:
        """Sample admission type with realistic weights per CLIF 2.1.0 schema."""
        weights = [0.50, 0.10, 0.10, 0.15, 0.10, 0.05]  # ed, facility, osh, direct, elective, other
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        return self.rng.choice(self.ADMISSION_TYPE_CATEGORIES, p=weights)

    def _sample_discharge_category(self) -> str:
        """Sample non-death discharge category per CLIF 2.1.0 schema."""
        # Weights for non-expired discharges (excluding "Expired" at index 2)
        non_expired_cats = [c for c in self.DISCHARGE_CATEGORIES if c != "Expired"]
        # One weight per non-expired category (16 total):
        # Home, SNF, Rehab, Hospice, LTACH, Acute Care, Group Home,
        # Chem Dep, AMA, Assisted Living, Still Admitted, Missing,
        # Other, Psychiatric, Shelter, Jail
        weights = [0.55, 0.12, 0.05, 0.05, 0.03, 0.02, 0.02, 0.01, 0.04, 0.03, 0.01, 0.04, 0.02, 0.005, 0.005, 0.005]
        weights = np.array(weights[:len(non_expired_cats)], dtype=float)
        weights /= weights.sum()
        return self.rng.choice(non_expired_cats, p=weights)

    def _get_discharge_name(self, discharge_category: str) -> str:
        """Convert discharge category to a free-text discharge name."""
        return discharge_category.lower().replace(" ", "_")
