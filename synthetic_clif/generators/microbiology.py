"""Microbiology culture and susceptibility generators."""

from datetime import datetime, timedelta, timezone
from typing import Optional
import uuid

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader


class MicrobiologyCultureGenerator(BaseGenerator):
    """Generate synthetic microbiology culture data.

    Creates microbiology_culture table with:
    - ~30% of patients have positive cultures
    - Realistic organism distribution
    - Proper timestamp ordering: order < collect < result
    """

    # Organism probabilities by fluid type (CLIF 2.1.0 snake_case values)
    ORGANISM_BY_FLUID = {
        "blood_buffy": {
            "organisms": [
                "staphylococcus_aureus",
                "escherichia_coli",
                "klebsiella_pneumoniae",
                "enterococcus_faecalis",
                "candida_albicans",
                "pseudomonas_aeruginosa",
                "no_growth",
            ],
            "weights": [0.20, 0.15, 0.10, 0.08, 0.05, 0.07, 0.35],
        },
        "genito_urinary_tract": {
            "organisms": [
                "escherichia_coli",
                "klebsiella_pneumoniae",
                "enterococcus_faecalis",
                "pseudomonas_aeruginosa",
                "proteus_mirabilis",
                "candida_albicans",
                "no_growth",
            ],
            "weights": [0.30, 0.12, 0.10, 0.08, 0.08, 0.07, 0.25],
        },
        "respiratory_tract": {
            "organisms": [
                "staphylococcus_aureus",
                "pseudomonas_aeruginosa",
                "klebsiella_pneumoniae",
                "acinetobacter_baumannii",
                "streptococcus_pneumoniae",
                "no_growth",
            ],
            "weights": [0.25, 0.18, 0.12, 0.08, 0.10, 0.27],
        },
        "woundsite": {
            "organisms": [
                "staphylococcus_aureus",
                "pseudomonas_aeruginosa",
                "escherichia_coli",
                "enterobacter_cloacae",
                "no_growth",
            ],
            "weights": [0.35, 0.15, 0.12, 0.08, 0.30],
        },
    }

    # CLIF 2.1.0 fluid categories for sampling
    FLUID_CATEGORIES = ["blood_buffy", "genito_urinary_tract", "respiratory_tract", "woundsite"]
    FLUID_WEIGHTS = [0.4, 0.3, 0.2, 0.1]

    # Organism to group mapping (CLIF 2.1.0 organism_group values)
    ORGANISM_GROUPS = {
        "staphylococcus_aureus": "staphylococcus_coag_pos",
        "escherichia_coli": "escherichia",
        "klebsiella_pneumoniae": "klebsiella",
        "pseudomonas_aeruginosa": "pseudomonas_wo_cepacia_maltophilia",
        "enterococcus_faecalis": "enterococcus",
        "enterococcus_faecium": "enterococcus",
        "candida_albicans": "candida_albicans",
        "candida_glabrata": "candida_nos",
        "acinetobacter_baumannii": "acinetobacter",
        "streptococcus_pneumoniae": "streptococcus",
        "enterobacter_cloacae": "enterobacter",
        "proteus_mirabilis": "other_organism",
    }

    METHOD_CATEGORIES = ["culture", "gram_stain", "smear"]
    METHOD_WEIGHTS = [0.80, 0.15, 0.05]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        culture_rate: float = 0.5,
        positive_rate: float = 0.3,
    ) -> pd.DataFrame:
        """Generate microbiology culture data.

        Args:
            hospitalizations_df: Hospitalization table DataFrame
            culture_rate: Proportion of hospitalizations with cultures
            positive_rate: Proportion of cultures that are positive

        Returns:
            DataFrame with microbiology_culture columns
        """
        records = []

        for _, hosp in hospitalizations_df.iterrows():
            hosp_id = hosp["hospitalization_id"]
            patient_id = hosp.get("patient_id", hosp_id)
            admit_time = hosp["admission_dttm"]
            discharge_time = hosp["discharge_dttm"]

            if pd.isna(admit_time):
                continue

            if pd.isna(discharge_time):
                discharge_time = admit_time + timedelta(days=5)

            # Determine if this hospitalization has cultures
            if self.rng.random() > culture_rate:
                continue

            hosp_cultures = self._generate_hospitalization_cultures(
                hosp_id, patient_id, admit_time, discharge_time, positive_rate
            )
            records.extend(hosp_cultures)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)
            df["collect_dttm"] = pd.to_datetime(df["collect_dttm"], utc=True)
            df["result_dttm"] = pd.to_datetime(df["result_dttm"], utc=True)

        return df

    def _generate_hospitalization_cultures(
        self,
        hospitalization_id: str,
        patient_id: str,
        admit_time: datetime,
        discharge_time: datetime,
        positive_rate: float,
    ) -> list[dict]:
        """Generate cultures for one hospitalization."""
        records = []
        los_hours = (discharge_time - admit_time).total_seconds() / 3600

        # Number of culture sets (1-4 depending on LOS)
        n_sets = min(4, max(1, int(los_hours / 48) + 1))

        for _ in range(n_sets):
            # Culture timing (usually early in admission or with fever)
            hours_from_admit = self.rng.uniform(0, min(72, los_hours))
            order_time = admit_time + timedelta(hours=hours_from_admit)

            if order_time >= discharge_time:
                continue

            # Sample fluid type (CLIF 2.1.0 categories)
            fluid = self.rng.choice(
                self.FLUID_CATEGORIES,
                p=self.FLUID_WEIGHTS,
            )

            # Generate culture ID
            culture_id = str(uuid.uuid4())[:8]

            # Determine organism
            is_positive = self.rng.random() < positive_rate
            if is_positive:
                organism_data = self.ORGANISM_BY_FLUID.get(
                    fluid, self.ORGANISM_BY_FLUID["blood_buffy"]
                )
                # Exclude "no_growth" for positive cultures
                organisms = [
                    o for o in organism_data["organisms"] if o != "no_growth"
                ]
                weights = organism_data["weights"][:-1]
                weights = np.array(weights) / sum(weights)
                organism = self.rng.choice(organisms, p=weights)
                organism_id = str(uuid.uuid4())[:8]
                organism_group = self.ORGANISM_GROUPS.get(organism, "Other")
            else:
                organism = "no_growth"
                organism_id = culture_id  # use culture_id as organism_id for no_growth
                organism_group = "no_growth"

            # Generate timestamps
            collect_delay = int(self.rng.integers(15, 60))  # minutes
            result_delay = int(self.rng.integers(24, 72))  # hours for cultures

            collect_time = order_time + timedelta(minutes=collect_delay)
            result_time = collect_time + timedelta(hours=result_delay)

            method = self.rng.choice(
                self.METHOD_CATEGORIES, p=self.METHOD_WEIGHTS
            )

            # LOINC codes by fluid type
            loinc_by_fluid = {
                "blood_buffy": "634-6",
                "genito_urinary_tract": "630-4",
                "respiratory_tract": "6463-4",
                "woundsite": "6462-6",
            }

            records.append(
                {
                    "patient_id": patient_id,
                    "hospitalization_id": hospitalization_id,
                    "organism_id": organism_id,
                    "order_dttm": order_time,
                    "collect_dttm": collect_time,
                    "result_dttm": result_time,
                    "fluid_category": fluid,
                    "fluid_name": self.name_from_category(fluid),
                    "method_category": method,
                    "method_name": self.name_from_category(method),
                    "organism_category": organism if organism != "no_growth" else "no_growth",
                    "organism_name": self.name_from_category(organism),
                    "organism_group": organism_group,
                    "lab_loinc_code": loinc_by_fluid.get(fluid, ""),
                }
            )

        return records


class MicrobiologySusceptibilityGenerator(BaseGenerator):
    """Generate synthetic susceptibility data for positive cultures.

    Creates microbiology_susceptibility table with:
    - Linked by organism_id to cultures
    - Realistic resistance patterns
    """

    # Susceptibility patterns by organism (CLIF 2.1.0 snake_case names)
    SUSCEPTIBILITY_PATTERNS = {
        "staphylococcus_aureus": {
            "antibiotics": ["Oxacillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.7, 1.0, 0.98, 0.99],
        },
        "escherichia_coli": {
            "antibiotics": [
                "Ampicillin",
                "Ceftriaxone",
                "Ciprofloxacin",
                "Meropenem",
                "Piperacillin-Tazobactam",
            ],
            "susceptible_rates": [0.5, 0.85, 0.75, 0.98, 0.90],
        },
        "klebsiella_pneumoniae": {
            "antibiotics": [
                "Ampicillin",
                "Ceftriaxone",
                "Ciprofloxacin",
                "Meropenem",
                "Piperacillin-Tazobactam",
            ],
            "susceptible_rates": [0.0, 0.80, 0.85, 0.95, 0.85],
        },
        "pseudomonas_aeruginosa": {
            "antibiotics": [
                "Cefepime",
                "Ciprofloxacin",
                "Meropenem",
                "Piperacillin-Tazobactam",
                "Gentamicin",
            ],
            "susceptible_rates": [0.85, 0.80, 0.85, 0.88, 0.90],
        },
        "enterococcus_faecalis": {
            "antibiotics": ["Ampicillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.95, 0.95, 0.98, 0.99],
        },
        "enterococcus_faecium": {
            "antibiotics": ["Ampicillin", "Vancomycin", "Daptomycin", "Linezolid"],
            "susceptible_rates": [0.1, 0.70, 0.95, 0.98],
        },
        "acinetobacter_baumannii": {
            "antibiotics": [
                "Ampicillin-Sulbactam",
                "Meropenem",
                "Ciprofloxacin",
                "Gentamicin",
                "Colistin",
            ],
            "susceptible_rates": [0.60, 0.70, 0.50, 0.75, 0.95],
        },
    }

    def generate(
        self,
        cultures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate susceptibility data for positive cultures.

        Args:
            cultures_df: Microbiology culture table DataFrame

        Returns:
            DataFrame with microbiology_susceptibility columns
        """
        records = []

        # No cultures (tiny datasets): the culture frame has no columns
        if "organism_id" not in cultures_df.columns:
            return pd.DataFrame(records)

        # Filter to positive cultures with organism_id
        positive_cultures = cultures_df[cultures_df["organism_id"].notna()]

        for _, culture in positive_cultures.iterrows():
            organism_id = culture["organism_id"]
            organism = culture["organism_category"]

            if organism is None:
                continue

            pattern = self.SUSCEPTIBILITY_PATTERNS.get(organism)
            if pattern is None:
                continue

            # Generate susceptibility for each antibiotic
            for abx, sus_rate in zip(
                pattern["antibiotics"], pattern["susceptible_rates"]
            ):
                # Determine susceptibility (CLIF 2.1.0 values)
                if self.rng.random() < sus_rate:
                    susceptibility = "susceptible"
                elif self.rng.random() < 0.3:
                    susceptibility = "indeterminate"
                else:
                    susceptibility = "non_susceptible"

                susceptibility_name = self.name_from_category(susceptibility)
                records.append(
                    {
                        "organism_id": organism_id,
                        "antimicrobial_name": abx,
                        "antimicrobial_category": abx.lower().replace("-", "_").replace(" ", "_"),
                        "susceptibility_category": susceptibility,
                        "sensitivity_name": susceptibility_name,
                        "susceptibility_name": susceptibility_name,
                    }
                )

        return pd.DataFrame(records)
