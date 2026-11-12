"""mCIDE (minimum Common ICU Data Elements) loader and manager."""

import os
from functools import cached_property
from pathlib import Path
from typing import Optional
import urllib.request
import zipfile
import tempfile
import shutil

import pandas as pd

from synthetic_clif.config.lab_catalog import CLIF21_LAB_CATEGORIES, LABS


class MCIDELoader:
    """Load and manage mCIDE permissible values from CSV files.

    mCIDE defines the standardized category values used in CLIF tables.
    This class loads these values from local CSV files or can fetch them
    from the CLIF GitHub repository.

    Example:
        >>> mcide = MCIDELoader()
        >>> mcide.vital_categories
        ['temp_c', 'heart_rate', 'sbp', 'dbp', 'spo2', 'respiratory_rate', ...]
        >>> mcide.get_category('discharge')
        ['Home', 'SNF', 'Expired', 'Hospice', ...]
    """

    GITHUB_RAW_BASE = "https://raw.githubusercontent.com/clif-consortium/mCIDE/main"

    def __init__(self, mcide_dir: Optional[Path] = None):
        """Initialize the mCIDE loader.

        Args:
            mcide_dir: Path to directory containing mCIDE CSV files.
                      If None, uses bundled default or embedded values.
        """
        self.mcide_dir = mcide_dir
        self._cache: dict[str, pd.DataFrame] = {}
        self._embedded_values = self._get_embedded_values()

    def _default_mcide_dir(self) -> Path:
        """Return path to bundled mCIDE directory."""
        return Path(__file__).parent / "mcide"

    def _get_embedded_values(self) -> dict[str, list[str]]:
        """Return embedded mCIDE values for common categories.

        These values are embedded in the code for reliability when
        external CSV files are not available. Based on CLIF 2.1.0.
        """
        return {
            # Patient demographics
            "sex": ["Female", "Male", "Other", "Unknown"],
            "race": [
                "American Indian or Alaska Native",
                "Asian",
                "Black or African American",
                "Native Hawaiian or Other Pacific Islander",
                "White",
                "Other",
                "Unknown",
            ],
            "ethnicity": [
                "Hispanic",
                "Non-Hispanic",
                "Unknown",
            ],
            # Hospitalization (CLIF 2.1.0)
            "admission_type": [
                "ed",
                "facility",
                "osh",
                "direct",
                "elective",
                "other",
            ],
            "discharge": [
                "Home",
                "Home Health",
                "SNF",
                "Expired",
                "Hospice",
                "LTACH",
                "Acute Care Hospital",
                "AMA",
                "Rehab",
                "Psych",
                "Jail",
                "Homeless",
                "Left AMA",
                "Other",
                "Unknown",
                "Still In",
                "Other Facility",
            ],
            # ADT — location_category permissible values per CLIF 2.1.0 spec
            "location": [
                "ed",
                "ward",
                "stepdown",
                "icu",
                "procedural",
                "l&d",
                "hospice",
                "psych",
                "rehab",
                "radiology",
                "dialysis",
                "other",
            ],
            # Vitals
            "vital": [
                "temp_c",
                "heart_rate",
                "sbp",
                "dbp",
                "spo2",
                "respiratory_rate",
                "map",
                "height_cm",
                "weight_kg",
            ],
            "vital_meas_site": [
                "Arterial",
                "Venous",
                "Oral",
                "Rectal",
                "Axillary",
                "Tympanic",
                "Temporal",
                "Bladder",
                "Skin",
                "Other",
            ],
            # Respiratory support
            "respiratory_device": [
                "IMV",
                "NIPPV",
                "CPAP",
                "High Flow NC",
                "Face Mask",
                "Trach Collar",
                "Nasal Cannula",
                "Room Air",
                "Other",
            ],
            "respiratory_mode": [
                "Assist Control-Volume Control",
                "Pressure Control",
                "Pressure-Regulated Volume Control",
                "SIMV",
                "Pressure Support/CPAP",
                "Volume Support",
                "Blow by",
                "CPAP",
                "BiPAP",
                "Other",
            ],
            # Labs: the 52 CLIF 2.1.0 categories (see config/lab_catalog.py)
            "lab": list(CLIF21_LAB_CATEGORIES),
            "lab_type": ["Routine", "STAT", "Timed", "Point of Care"],
            "lab_order": ["bmp", "cbc", "coags", "blood_gas", "lft", "misc"],
            # Medications
            "medication": [
                "norepinephrine",
                "epinephrine",
                "vasopressin",
                "dopamine",
                "dobutamine",
                "phenylephrine",
                "milrinone",
                "propofol",
                "dexmedetomidine",
                "midazolam",
                "fentanyl",
                "morphine",
                "hydromorphone",
                "ketamine",
                "rocuronium",
                "cisatracurium",
                "vecuronium",
                "heparin",
                "insulin",
                "vancomycin",
                "piperacillin_tazobactam",
                "cefepime",
                "meropenem",
                "other",
            ],
            "med_route": [
                "iv",
                "po",
                "im",
                "sc",
                "inhaled",
                "topical",
                "pr",
                "sl",
                "enteral",
                "other",
            ],
            "mar_action": [
                "given",
                "bolus",
                "not_given",
                "held",
                "refused",
                "other",
            ],
            # Microbiology (CLIF 2.1.0 snake_case values)
            "culture_fluid": [
                "blood_buffy",
                "genito_urinary_tract",
                "respiratory_tract",
                "respiratory_tract_lower",
                "woundsite",
                "meninges_csf",
                "peritoneum",
                "pleural_cavity_fluid",
                "catheter_tip",
                "other_unspecified",
            ],
            "organism": [
                "staphylococcus_aureus",
                "escherichia_coli",
                "klebsiella_pneumoniae",
                "pseudomonas_aeruginosa",
                "enterococcus_faecalis",
                "enterococcus_faecium",
                "candida_albicans",
                "candida_glabrata",
                "acinetobacter_baumannii",
                "streptococcus_pneumoniae",
                "enterobacter_cloacae",
                "proteus_mirabilis",
                "no_growth",
            ],
            "organism_group": [
                "staphylococcus_coag_pos",
                "staphylococcus_coag_neg",
                "enterococcus",
                "streptococcus",
                "escherichia",
                "klebsiella",
                "pseudomonas_wo_cepacia_maltophilia",
                "acinetobacter",
                "enterobacter",
                "candida_albicans",
                "candida_nos",
                "no_growth",
                "other_organism",
            ],
            "antibiotic": [
                "vancomycin",
                "ceftriaxone",
                "cefepime",
                "meropenem",
                "piperacillin_tazobactam",
                "ciprofloxacin",
                "levofloxacin",
                "gentamicin",
                "ampicillin",
                "oxacillin",
                "daptomycin",
                "linezolid",
                "colistin",
                "ampicillin_sulbactam",
            ],
            "susceptibility": [
                "susceptible",
                "intermediate",
                "non_susceptible",
            ],
            # Assessments (CLIF 2.1.0)
            "assessment": [
                "gcs_total",
                "gcs_eye",
                "gcs_verbal",
                "gcs_motor",
                "RASS",
                "cam_total",
                "NRS",
                "braden_total",
                "Morse Fall Scale",
                "apache_ii",
                "sofa",
                "other",
            ],
            # Procedures
            "procedure": [
                "Central Line Insertion",
                "Arterial Line Insertion",
                "Intubation",
                "Extubation",
                "Bronchoscopy",
                "Chest Tube Insertion",
                "Paracentesis",
                "Thoracentesis",
                "Lumbar Puncture",
                "Dialysis Catheter Insertion",
                "PICC Line Insertion",
                "Tracheostomy",
                "Other",
            ],
            # Diagnosis (CLIF 2.1.0)
            "poa": [0, 1],
            # Code status (CLIF 2.1.0)
            "code_status": [
                "DNR",
                "DNAR",
                "UDNR",
                "DNR/DNI",
                "DNAR/DNI",
                "AND",
                "Full",
                "Presume Full",
                "Other",
            ],
            # Position (CLIF 2.1.0)
            "position": [
                "prone",
                "not_prone",
            ],
            # CRRT (CLIF 2.1.0)
            "crrt_mode": ["scuf", "cvvh", "cvvhd", "cvvhdf", "avvh"],
            # Concept tables
            "ecmo_device": ["ECMO", "VAD", "IABP", "Impella", "Other"],
            "ecmo_config": ["VV", "VA", "VAV", "Other"],
            "io": [
                "IV Fluids",
                "Blood Products",
                "Enteral Nutrition",
                "PO Intake",
                "Urine Output",
                "Drain Output",
                "Stool",
                "NG Output",
                "Other",
            ],
            "hemodynamic": [
                "CVP",
                "PA Systolic",
                "PA Diastolic",
                "PA Mean",
                "PCWP",
                "Cardiac Output",
                "Cardiac Index",
                "SVR",
                "PVR",
                "SvO2",
            ],
            "icu_order": [
                "PT Evaluation",
                "OT Evaluation",
                "Speech Evaluation",
                "Nutrition Consult",
                "Palliative Care Consult",
                "Social Work Consult",
                "Other",
            ],
            "micro_nonculture_test": [
                "COVID-19 PCR",
                "Influenza PCR",
                "RSV PCR",
                "Respiratory Viral Panel",
                "C. diff Toxin",
                "Procalcitonin",
                "Other",
            ],
            "micro_result": ["Positive", "Negative", "Indeterminate"],
            "provider_role": [
                "Attending",
                "Resident",
                "Fellow",
                "NP",
                "PA",
                "RN",
                "Pharmacist",
                "Respiratory Therapist",
                "Physical Therapist",
                "Other",
            ],
            "therapy": [
                "Physical Therapy",
                "Occupational Therapy",
                "Speech Therapy",
                "Respiratory Therapy",
                "Other",
            ],
            "blood_product": [
                "Packed RBCs",
                "Fresh Frozen Plasma",
                "Platelets",
                "Cryoprecipitate",
                "Whole Blood",
                "Other",
            ],
        }

    def get_category(self, category: str) -> list[str]:
        """Get permissible values for a category.

        Args:
            category: The mCIDE category name

        Returns:
            List of valid values for this category
        """
        # First try embedded values
        if category in self._embedded_values:
            return self._embedded_values[category].copy()

        # Try to load from CSV if directory specified
        if self.mcide_dir and self.mcide_dir.exists():
            df = self._load_category_csv(category)
            if df is not None and "category" in df.columns:
                return df["category"].dropna().tolist()

        # Return empty list if not found
        return []

    def _load_category_csv(self, category: str) -> Optional[pd.DataFrame]:
        """Load a category CSV file."""
        if category in self._cache:
            return self._cache[category]

        # Search for matching CSV file
        search_patterns = [
            f"**/clif_{category}_categories.csv",
            f"**/{category}_categories.csv",
            f"**/clif_{category}.csv",
            f"**/{category}.csv",
        ]

        for pattern in search_patterns:
            files = list(self.mcide_dir.glob(pattern))
            if files:
                try:
                    df = pd.read_csv(files[0])
                    self._cache[category] = df
                    return df
                except Exception:
                    pass

        return None

    def fetch_from_github(self, version: str = "main") -> Path:
        """Download mCIDE CSVs from the CLIF GitHub repository.

        Args:
            version: Git branch/tag to download from

        Returns:
            Path to downloaded mCIDE directory
        """
        url = f"https://github.com/clif-consortium/mCIDE/archive/refs/heads/{version}.zip"

        # Create destination directory
        dest_dir = self._default_mcide_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "mcide.zip"

            # Download zip file
            urllib.request.urlretrieve(url, zip_path)

            # Extract
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find extracted directory and copy contents
            extracted = list(Path(tmpdir).glob("mCIDE-*"))
            if extracted:
                for item in extracted[0].iterdir():
                    dest = dest_dir / item.name
                    if item.is_dir():
                        if dest.exists():
                            shutil.rmtree(dest)
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)

        # Update mcide_dir to use downloaded files
        self.mcide_dir = dest_dir
        self._cache.clear()

        return dest_dir

    # Convenience properties for common categories

    @cached_property
    def vital_categories(self) -> list[str]:
        """Vital sign categories."""
        return self.get_category("vital")

    @cached_property
    def lab_categories(self) -> list[str]:
        """Lab test categories."""
        return self.get_category("lab")

    @cached_property
    def device_categories(self) -> list[str]:
        """Respiratory device categories."""
        return self.get_category("respiratory_device")

    @cached_property
    def medication_categories(self) -> list[str]:
        """Medication categories."""
        return self.get_category("medication")

    @cached_property
    def location_categories(self) -> list[str]:
        """ADT location categories."""
        return self.get_category("location")

    @cached_property
    def discharge_categories(self) -> list[str]:
        """Discharge disposition categories."""
        return self.get_category("discharge")

    @cached_property
    def sex_categories(self) -> list[str]:
        """Sex categories."""
        return self.get_category("sex")

    @cached_property
    def race_categories(self) -> list[str]:
        """Race categories."""
        return self.get_category("race")

    @cached_property
    def ethnicity_categories(self) -> list[str]:
        """Ethnicity categories."""
        return self.get_category("ethnicity")

    def get_lab_reference_units(self) -> dict[str, str]:
        """CLIF 2.1.0 reference units for the lab categories (mCIDE)."""
        return {k: v.unit_21 for k, v in LABS.items() if v.unit_21 is not None}

    def get_lab_reference_ranges(self) -> dict[str, tuple[float, float]]:
        """Get normal reference ranges for lab categories."""
        return {
            "sodium": (136, 145),
            "potassium": (3.5, 5.0),
            "chloride": (98, 106),
            "bicarbonate": (22, 28),
            "bun": (7, 20),
            "creatinine": (0.7, 1.3),
            "glucose_serum": (70, 100),
            "calcium_total": (8.5, 10.5),
            "magnesium": (1.7, 2.3),
            "phosphate": (2.5, 4.5),
            "albumin": (3.5, 5.0),
            "total_protein": (6.0, 8.0),
            "ast": (10, 40),
            "alt": (10, 40),
            "alkaline_phosphatase": (40, 130),
            "bilirubin_total": (0.1, 1.2),
            "bilirubin_conjugated": (0, 0.3),
            "lactate": (0.5, 2.0),
            "hemoglobin": (12, 17),
            "wbc": (4.5, 11.0),
            "platelet_count": (150, 400),
            "inr": (0.9, 1.1),
            "pt": (11, 14),
            "ptt": (25, 35),
            "fibrinogen": (200, 400),
            "d_dimer": (0, 500),
            "troponin_i": (0, 26),
            "troponin_t": (0, 14),
            "crp": (0, 10),
            "procalcitonin": (0, 0.5),
            "ph_arterial": (7.35, 7.45),
            "ph_venous": (7.31, 7.41),
            "pco2_arterial": (35, 45),
            "pco2_venous": (41, 51),
            "po2_arterial": (80, 100),
            "so2_central_venous": (60, 80),
        }
