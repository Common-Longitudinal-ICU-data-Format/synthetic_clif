"""Microbiology culture and susceptibility generators."""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.physiology.course import course_for_row


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
        "staphylococcus_epidermidis": "staphylococcus_coag_neg",
    }
    GRAM_POS = {"staphylococcus_aureus", "enterococcus_faecalis", "streptococcus_pneumoniae"}
    COURSE_FLUID = {"blood": "blood_buffy", "urine": "genito_urinary_tract",
                    "respiratory": "respiratory_tract", "wound": "woundsite"}
    LOINC = {"blood_buffy": "634-6", "genito_urinary_tract": "630-4",
             "respiratory_tract": "6463-4", "woundsite": "6462-6"}

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

            course = course_for_row(hosp)
            if course is None:
                continue
            hosp_cultures = self._course_cultures(hosp_id, patient_id, admit_time, discharge_time, course)
            records.extend(hosp_cultures)

        df = pd.DataFrame(records)

        if len(df) > 0:
            df["order_dttm"] = pd.to_datetime(df["order_dttm"], utc=True)
            df["collect_dttm"] = pd.to_datetime(df["collect_dttm"], utc=True)
            df["result_dttm"] = pd.to_datetime(df["result_dttm"], utc=True)

        return df

    def _hex_id(self) -> str:
        return f"{int(self.rng.integers(0, 16**8)):08x}"

    def _course_cultures(self, hosp_id, patient_id, admit_time, discharge_time, course) -> list[dict]:
        """Cultures drawn when the course recognises infection, or for fever.

        The infecting organism is chosen once per stay to match the course's
        pathogen (gram-positive or gram-negative) and reappears in every
        culture from an infected site; contaminants are coagulase-negative
        staphylococci in a single blood culture set.
        """
        rows = []
        pathogen = {}
        for cx in course.cultures:
            fluid = self.COURSE_FLUID[cx["fluid"]]
            order_time = admit_time + timedelta(hours=float(cx["h"]))
            if order_time >= discharge_time:
                continue
            collect_time = order_time + timedelta(minutes=int(self.rng.integers(10, 60)))
            if not cx["positive"]:
                organism = "no_growth"
            elif cx.get("contaminant"):
                organism = "staphylococcus_epidermidis"
            else:
                options = [o for o in self.ORGANISM_BY_FLUID[fluid]["organisms"] if o != "no_growth"]
                weights = np.array([w for o, w in zip(self.ORGANISM_BY_FLUID[fluid]["organisms"],
                                                      self.ORGANISM_BY_FLUID[fluid]["weights"]) if o != "no_growth"])
                gram = course.pathogen_gram
                if course.infected and gram:
                    match = [(o, w) for o, w in zip(options, weights)
                             if (o in self.GRAM_POS) == (gram == "gram_pos")]
                    if match:
                        options, weights = map(list, zip(*match))
                        weights = np.array(weights)
                known = [o for o in pathogen.values() if o in options]
                organism = known[0] if known else str(self.rng.choice(options, p=weights / weights.sum()))
                pathogen[fluid] = organism
            if organism == "no_growth":
                hours = {"blood_buffy": 120, "genito_urinary_tract": 36, "respiratory_tract": 60}.get(fluid, 72)
                result_time = collect_time + timedelta(hours=hours + self.rng.uniform(-6, 6))
                organism_id = self._hex_id()
                group = "no_growth"
            else:
                result_time = collect_time + timedelta(hours=self.rng.uniform(12, 48))
                organism_id = self._hex_id()
                group = self.ORGANISM_GROUPS.get(organism, "other_organism")
            method = "culture" if self.rng.random() < 0.9 else "gram_stain"
            rows.append({
                "patient_id": patient_id,
                "hospitalization_id": hosp_id,
                "organism_id": organism_id,
                "order_dttm": order_time,
                "collect_dttm": collect_time,
                "result_dttm": result_time,
                "fluid_category": fluid,
                "fluid_name": self.name_from_category(fluid),
                "method_category": method,
                "method_name": self.name_from_category(method),
                "organism_category": organism,
                "organism_name": self.name_from_category(organism),
                "organism_group": group,
                "lab_loinc_code": self.LOINC.get(fluid, ""),
            })
        return rows

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
            culture_id = self._hex_id()

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
                organism_id = self._hex_id()
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
