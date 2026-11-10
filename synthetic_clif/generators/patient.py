"""Patient table generator."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.generators.hospitalization import CLIF_DATE_START, CLIF_DATE_END


class PatientGenerator(BaseGenerator):
    """Generate synthetic patient demographics.

    Creates patient table with:
    - patient_id (UUID format)
    - sex_category, race_category, ethnicity_category, language_category (mCIDE categories)
    - birth_date (realistic age distribution 18-95)
    - death_dttm (~15% mortality, correlated with hospitalizations)
    """

    # Language categories per CLIF 2.1.0 schema
    LANGUAGE_CATEGORIES = [
        "English",
        "Spanish",
        "French",
        "Haitian Creole",
        "Italian",
        "Portuguese",
        "German",
        "Chinese",
        "Vietnamese",
        "Korean",
        "Tagalog",
        "Arabic",
        "Russian",
        "Sign Language",
        "Unknown or NA",
    ]
    # Weights roughly based on US demographics
    LANGUAGE_WEIGHTS = [0.78, 0.12, 0.01, 0.005, 0.005, 0.01, 0.005, 0.02, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.02]

    def __init__(
        self,
        seed: Optional[int] = None,
        mcide: Optional[MCIDELoader] = None,
    ):
        super().__init__(seed, mcide)
        self.faker = Faker()
        Faker.seed(seed)

    def generate(
        self,
        n_patients: int,
        mortality_rate: float = 0.258,
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate patient demographics.

        Args:
            n_patients: Number of patients to generate
            mortality_rate: Proportion of patients who die (0-1)
            reference_date: Reference date for age/death calculations

        Returns:
            DataFrame with patient table columns
        """
        if reference_date is None:
            reference_date = CLIF_DATE_END  # fixed, so the same seed gives the same birth dates

        # Generate patient IDs
        patient_ids = self.generate_uuids(n_patients)

        # Generate demographics
        # Consortium aggregate: Male 54.5%, Female 45.5%, Other <0.1%, Unknown <0.1%
        sex_weights = [0.455, 0.545, 0.0001, 0.0001]  # Female, Male, Other, Unknown
        sex_categories = self.sample_category("sex", n_patients, sex_weights)

        # Consortium aggregate: AI/AN 0.5%, Asian 4.5%, Black 19.5%, NHPI 0.3%,
        # White 63.3%, Other 5.2%, Unknown 6.6%
        race_weights = [0.005, 0.045, 0.195, 0.003, 0.633, 0.052, 0.066]
        race_categories = self.sample_category("race", n_patients, race_weights)

        # Consortium aggregate: Hispanic 5.8%, Non-Hispanic 86.7%, Unknown 7.4%
        ethnicity_weights = [0.058, 0.867, 0.074]
        ethnicity_categories = self.sample_category("ethnicity", n_patients, ethnicity_weights)

        # Generate language categories
        language_weights = np.array(self.LANGUAGE_WEIGHTS[:len(self.LANGUAGE_CATEGORIES)], dtype=float)
        language_weights /= language_weights.sum()
        language_categories = self.rng.choice(
            self.LANGUAGE_CATEGORIES, size=n_patients, p=language_weights
        ).tolist()

        # Generate birth dates (age distribution typical for ICU)
        # Bimodal: younger trauma/surgical, older medical
        ages = self._generate_age_distribution(n_patients)
        birth_dates = [
            (reference_date - timedelta(days=int(age * 365.25))).date() for age in ages
        ]

        # Generate death dates for those who die
        n_deaths = int(n_patients * mortality_rate)
        death_indices = self.rng.choice(n_patients, size=n_deaths, replace=False)
        death_dttms = [None] * n_patients

        total_clif_seconds = int((CLIF_DATE_END - CLIF_DATE_START).total_seconds())
        for idx in death_indices:
            offset_seconds = int(self.rng.integers(0, total_clif_seconds))
            death_dttms[idx] = CLIF_DATE_START + timedelta(seconds=offset_seconds)

        # Create DataFrame with columns ordered per CLIF 2.1.0 schema
        df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "birth_date": birth_dates,
                "death_dttm": death_dttms,
                "race_category": race_categories,
                "ethnicity_category": ethnicity_categories,
                "sex_category": sex_categories,
                "language_category": language_categories,
            }
        )

        # Convert datetime columns
        df["birth_date"] = pd.to_datetime(df["birth_date"])
        df["death_dttm"] = pd.to_datetime(df["death_dttm"], utc=True)

        # Derive _name companion columns
        df["race_name"] = df["race_category"].apply(lambda x: self.name_from_category(x) if pd.notna(x) else None)
        df["sex_name"] = df["sex_category"].apply(lambda x: self.name_from_category(x) if pd.notna(x) else None)
        df["ethnicity_name"] = df["ethnicity_category"].apply(lambda x: self.name_from_category(x) if pd.notna(x) else None)
        df["language_name"] = df["language_category"]  # already readable

        # Add some missingness to demographics (rare)
        df = self.add_missingness(df, "race_category", 0.03)
        df = self.add_missingness(df, "ethnicity_category", 0.02)
        df = self.add_missingness(df, "language_category", 0.05)

        return df

    def _generate_age_distribution(self, n: int) -> np.ndarray:
        """Generate age distribution typical for ICU population.

        Consortium aggregate: median 66 [Q1=47, Q3=79].
        Uses mixture of distributions:
        - 15% younger (trauma, surgical): mean 38, std 10
        - 85% older (medical): mean 72, std 13

        Returns ages in years, bounded to [18, 100].
        """
        ages = np.zeros(n)

        # Young cohort (15%)
        n_young = int(n * 0.15)
        ages[:n_young] = self.rng.normal(38, 10, n_young)

        # Older cohort (85%)
        ages[n_young:] = self.rng.normal(72, 13, n - n_young)

        # Shuffle and bound
        self.rng.shuffle(ages)
        return np.clip(ages, 18, 100)
