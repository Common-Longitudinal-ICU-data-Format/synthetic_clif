"""Dataset orchestrator for generating complete synthetic CLIF datasets."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from synthetic_clif.config.mcide import MCIDELoader
from synthetic_clif.generators.patient import PatientGenerator
from synthetic_clif.generators.hospitalization import HospitalizationGenerator
from synthetic_clif.generators.adt import ADTGenerator
from synthetic_clif.generators.vitals import VitalsGenerator
from synthetic_clif.generators.labs import LabsGenerator
from synthetic_clif.generators.respiratory import RespiratoryGenerator
from synthetic_clif.generators.medications import (
    MedicationContinuousGenerator,
    MedicationIntermittentGenerator,
)
from synthetic_clif.generators.microbiology import (
    MicrobiologyCultureGenerator,
    MicrobiologySusceptibilityGenerator,
)
from synthetic_clif.generators.assessments import PatientAssessmentsGenerator
from synthetic_clif.generators.procedures import (
    PatientProceduresGenerator,
    HospitalDiagnosisGenerator,
)
from synthetic_clif.generators.other import (
    CodeStatusGenerator,
    PositionGenerator,
    CRRTTherapyGenerator,
)
from synthetic_clif.generators.concept import (
    ClinicalTrialGenerator,
    ECMOMCSGenerator,
    IntakeOutputGenerator,
    InvasiveHemodynamicsGenerator,
    KeyICUOrdersGenerator,
    MedicationOrdersGenerator,
    MicrobiologyNoncultureGenerator,
    PatientDiagnosisGenerator,
    PlaceBasedIndexGenerator,
    ProviderGenerator,
    TherapyDetailsGenerator,
    TransfusionGenerator,
)


SUPPORTED_CLIF_VERSIONS = ("2.1", "3.0")


def normalize_clif_version(version) -> str:
    """Accept "2.1", "2.1.0", "3", "3.0", "3.0.0" and return "2.1" or "3.0"."""
    v = str(version).strip().lower().lstrip("v")
    parts = v.split(".")
    short = ".".join((parts + ["0"])[:2])
    if short not in SUPPORTED_CLIF_VERSIONS:
        raise ValueError(
            f"Unsupported CLIF version {version!r}; choose one of {SUPPORTED_CLIF_VERSIONS}"
        )
    return short


class SyntheticCLIFDataset:
    """Orchestrates generation of complete synthetic CLIF datasets.

    This class coordinates all table generators to produce a coherent dataset
    with proper referential integrity and clinically realistic correlations.

    Example:
        >>> dataset = SyntheticCLIFDataset(n_patients=10, n_hospitalizations=12)
        >>> tables = dataset.generate()
        >>> dataset.to_parquet(Path("data/test/"))
    """

    def __init__(
        self,
        n_patients: int = 10,
        n_hospitalizations: int = 12,
        seed: int = 42,
        mcide_dir: Optional[Path] = None,
        include_concept_tables: bool = True,
        workers: int = 1,
        clif_version: str = "3.0",
    ):
        """Initialize the synthetic dataset generator.

        Args:
            n_patients: Number of patients to generate
            n_hospitalizations: Total number of hospitalizations
            seed: Random seed for reproducibility
            mcide_dir: Optional path to mCIDE CSV directory
            include_concept_tables: Whether to generate concept tables (draft status)
            workers: Number of parallel workers (1 = sequential, >1 = parallel big-3)
            clif_version: CLIF version of the output tables, "3.0" (default) or
                "2.1". 3.0 output is produced by converting the 2.1 simulation
                (see ``synthetic_clif.clif3``).
        """
        self.n_patients = n_patients
        self.n_hospitalizations = n_hospitalizations
        self.seed = seed
        self.mcide_dir = mcide_dir
        self.include_concept_tables = include_concept_tables
        self.workers = max(1, workers)
        self.clif_version = normalize_clif_version(clif_version)

        # Initialize shared mCIDE loader
        self.mcide = MCIDELoader(mcide_dir)

        # Initialize all generators with correlated seeds
        self._init_generators()

        # Storage for generated tables
        self._tables: dict[str, pd.DataFrame] = {}

    def _init_generators(self):
        """Initialize all table generators with correlated seeds."""
        # Use seed to create reproducible child seeds
        import numpy as np

        rng = np.random.default_rng(self.seed)

        def next_seed():
            return int(rng.integers(0, 2**31))

        # Beta table generators
        self.patient_gen = PatientGenerator(seed=next_seed(), mcide=self.mcide)
        self.hosp_gen = HospitalizationGenerator(seed=next_seed(), mcide=self.mcide)
        self.adt_gen = ADTGenerator(seed=next_seed(), mcide=self.mcide)
        self.vitals_gen = VitalsGenerator(seed=next_seed(), mcide=self.mcide)
        self.labs_gen = LabsGenerator(seed=next_seed(), mcide=self.mcide)
        self.respiratory_gen = RespiratoryGenerator(seed=next_seed(), mcide=self.mcide)
        self.med_continuous_gen = MedicationContinuousGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.med_intermittent_gen = MedicationIntermittentGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.culture_gen = MicrobiologyCultureGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.susceptibility_gen = MicrobiologySusceptibilityGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.assessments_gen = PatientAssessmentsGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.procedures_gen = PatientProceduresGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.diagnosis_gen = HospitalDiagnosisGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.code_status_gen = CodeStatusGenerator(seed=next_seed(), mcide=self.mcide)
        self.position_gen = PositionGenerator(seed=next_seed(), mcide=self.mcide)
        self.crrt_gen = CRRTTherapyGenerator(seed=next_seed(), mcide=self.mcide)

        # Concept table generators
        self.clinical_trial_gen = ClinicalTrialGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.ecmo_gen = ECMOMCSGenerator(seed=next_seed(), mcide=self.mcide)
        self.io_gen = IntakeOutputGenerator(seed=next_seed(), mcide=self.mcide)
        self.hemodynamics_gen = InvasiveHemodynamicsGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.icu_orders_gen = KeyICUOrdersGenerator(seed=next_seed(), mcide=self.mcide)
        self.med_orders_gen = MedicationOrdersGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.nonculture_gen = MicrobiologyNoncultureGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.patient_dx_gen = PatientDiagnosisGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.place_index_gen = PlaceBasedIndexGenerator(
            seed=next_seed(), mcide=self.mcide
        )
        self.provider_gen = ProviderGenerator(seed=next_seed(), mcide=self.mcide)
        self.therapy_gen = TherapyDetailsGenerator(seed=next_seed(), mcide=self.mcide)
        self.transfusion_gen = TransfusionGenerator(seed=next_seed(), mcide=self.mcide)

        # CLIF 3.0-only tables (seeded last so the 2.1 tables are unchanged)
        from synthetic_clif.generators import v3

        self.drain_gen = v3.DrainGenerator(seed=next_seed(), mcide=self.mcide)
        self.ihd_gen = v3.IntermittentDialysisGenerator(seed=next_seed(), mcide=self.mcide)
        self.attributes_gen = v3.PatientAttributesGenerator(seed=next_seed(), mcide=self.mcide)
        self.radiology_gen = v3.RadiologyGenerator(seed=next_seed(), mcide=self.mcide)
        self.notes_gen = v3.ClinicalNotesFactsGenerator(seed=next_seed(), mcide=self.mcide)
        self.validated_dx_gen = v3.ValidatedDiagnosisGenerator(seed=next_seed(), mcide=self.mcide)
        self.scores_gen = v3.ScoresGenerator(seed=next_seed(), mcide=self.mcide)

    def generate(self, verbose: bool = True) -> dict[str, pd.DataFrame]:
        """Generate all tables with referential integrity.

        Args:
            verbose: If True, print progress with timing information

        Returns:
            Dictionary mapping table names to DataFrames
        """
        import time

        def _log(msg: str, start_time: float = None, rows: int = None):
            if not verbose:
                return
            if start_time is not None:
                elapsed = time.time() - start_time
                row_info = f" ({rows:,} rows)" if rows is not None else ""
                print(f"  {msg}{row_info} [{elapsed:.2f}s]")
            else:
                print(msg)

        total_start = time.time()

        _log(f"Generating synthetic CLIF dataset...")
        _log(f"  Patients: {self.n_patients}")
        _log(f"  Hospitalizations: {self.n_hospitalizations}")
        _log(f"  Seed: {self.seed}")
        _log("")

        # Generate base tables
        t = time.time()
        patients = self.patient_gen.generate(self.n_patients)
        self._tables["patient"] = patients
        _log("patient", t, len(patients))

        t = time.time()
        # Stays are laid out first; who dies, and where survivors go, is then
        # decided from each stay's clinical course (synthetic_clif.physiology).
        no_deaths = patients.assign(death_dttm=pd.Series(pd.NaT, index=patients.index, dtype="datetime64[ns, UTC]"))
        hospitalizations = self.hosp_gen.generate(no_deaths, self.n_hospitalizations)
        patients, hospitalizations = self._assign_outcomes(patients, hospitalizations)
        self._tables["patient"] = patients
        self._tables["hospitalization"] = hospitalizations
        _log("hospitalization", t, len(hospitalizations))

        # Every clinical generator observes one shared course per stay
        # (synthetic_clif.physiology). The course seed and sex ride along on a
        # private copy of the hospitalization table; the published table is
        # unchanged.
        course_hosps = self._course_hospitalizations(hospitalizations, patients)

        t = time.time()
        adt = self.adt_gen.generate(course_hosps)
        self._tables["adt"] = adt
        _log("adt", t, len(adt))

        # Generate big-3 time-series tables (vitals, labs, respiratory)
        if self.workers > 1:
            self._generate_big3_parallel(course_hosps, adt, _log)
        else:
            t = time.time()
            vitals = self.vitals_gen.generate(course_hosps, adt)
            self._tables["vitals"] = vitals
            _log("vitals", t, len(vitals))

            t = time.time()
            labs = self.labs_gen.generate(course_hosps)
            self._tables["labs"] = labs
            _log("labs", t, len(labs))

            t = time.time()
            respiratory = self.respiratory_gen.generate(course_hosps)
            self._tables["respiratory_support"] = respiratory
            _log("respiratory_support", t, len(respiratory))

        # Generate remaining time-series tables (reference self._tables for deps)
        respiratory = self._tables["respiratory_support"]

        t = time.time()
        med_continuous = self.med_continuous_gen.generate(course_hosps, respiratory)
        self._tables["medication_admin_continuous"] = med_continuous
        _log("medication_admin_continuous", t, len(med_continuous))

        t = time.time()
        med_intermittent = self.med_intermittent_gen.generate(course_hosps)
        self._tables["medication_admin_intermittent"] = med_intermittent
        _log("medication_admin_intermittent", t, len(med_intermittent))

        t = time.time()
        cultures = self.culture_gen.generate(course_hosps)
        self._tables["microbiology_culture"] = cultures
        _log("microbiology_culture", t, len(cultures))

        t = time.time()
        susceptibilities = self.susceptibility_gen.generate(cultures)
        self._tables["microbiology_susceptibility"] = susceptibilities
        _log("microbiology_susceptibility", t, len(susceptibilities))

        t = time.time()
        assessments = self.assessments_gen.generate(course_hosps, respiratory)
        self._tables["patient_assessments"] = assessments
        _log("patient_assessments", t, len(assessments))

        t = time.time()
        procedures = self.procedures_gen.generate(course_hosps)
        self._tables["patient_procedures"] = procedures
        _log("patient_procedures", t, len(procedures))

        t = time.time()
        diagnoses = self.diagnosis_gen.generate(course_hosps)
        self._tables["hospital_diagnosis"] = diagnoses
        _log("hospital_diagnosis", t, len(diagnoses))

        t = time.time()
        code_status = self.code_status_gen.generate(course_hosps)
        self._tables["code_status"] = code_status
        _log("code_status", t, len(code_status))

        t = time.time()
        position = self.position_gen.generate(course_hosps, respiratory)
        self._tables["position"] = position
        _log("position", t, len(position))

        t = time.time()
        crrt = self.crrt_gen.generate(course_hosps)
        self._tables["crrt_therapy"] = crrt
        _log("crrt_therapy", t, len(crrt))

        # Generate concept tables if requested
        if self.include_concept_tables:
            _log("")
            _log("  Generating concept tables...")

            t = time.time()
            self._tables["clinical_trial"] = self.clinical_trial_gen.generate(
                course_hosps
            )
            _log("clinical_trial", t, len(self._tables["clinical_trial"]))

            t = time.time()
            self._tables["ecmo_mcs"] = self.ecmo_gen.generate(course_hosps)
            _log("ecmo_mcs", t, len(self._tables["ecmo_mcs"]))

            t = time.time()
            self._tables["intake_output"] = self.io_gen.generate(course_hosps)
            _log("intake_output", t, len(self._tables["intake_output"]))

            t = time.time()
            self._tables["invasive_hemodynamics"] = self.hemodynamics_gen.generate(
                course_hosps
            )
            _log("invasive_hemodynamics", t, len(self._tables["invasive_hemodynamics"]))

            t = time.time()
            self._tables["key_icu_orders"] = self.icu_orders_gen.generate(
                course_hosps
            )
            _log("key_icu_orders", t, len(self._tables["key_icu_orders"]))

            t = time.time()
            self._tables["medication_orders"] = self.med_orders_gen.generate(
                course_hosps, med_continuous, med_intermittent
            )
            _log("medication_orders", t, len(self._tables["medication_orders"]))

            t = time.time()
            self._tables["microbiology_nonculture"] = self.nonculture_gen.generate(
                course_hosps, include_influenza=self.clif_version == "3.0"
            )
            _log("microbiology_nonculture", t, len(self._tables["microbiology_nonculture"]))

            t = time.time()
            self._tables["patient_diagnosis"] = self.patient_dx_gen.generate(patients, course_hosps)
            _log("patient_diagnosis", t, len(self._tables["patient_diagnosis"]))

            t = time.time()
            self._tables["place_based_index"] = self.place_index_gen.generate(course_hosps)
            _log("place_based_index", t, len(self._tables["place_based_index"]))

            t = time.time()
            self._tables["provider"] = self.provider_gen.generate(course_hosps)
            _log("provider", t, len(self._tables["provider"]))

            t = time.time()
            self._tables["therapy_details"] = self.therapy_gen.generate(
                course_hosps, orders_df=self._tables.get("key_icu_orders")
            )
            _log("therapy_details", t, len(self._tables["therapy_details"]))

            t = time.time()
            self._tables["transfusion"] = self.transfusion_gen.generate(course_hosps)
            _log("transfusion", t, len(self._tables["transfusion"]))

        if self.clif_version == "3.0":
            native = self._generate_clif3_tables(course_hosps, patients, _log)
            t = time.time()
            from synthetic_clif.clif3 import convert_to_clif3

            self._tables_v21 = self._tables
            self._tables_v3_native = native
            self._tables = convert_to_clif3(self._tables, seed=self.seed, native_tables=native)
            _log("")
            _log("converted to CLIF 3.0", t, sum(len(df) for df in self._tables.values()))

        total_elapsed = time.time() - total_start
        total_rows = sum(len(df) for df in self._tables.values())
        _log("")
        _log(f"  Total: {total_rows:,} rows across {len(self._tables)} tables [{total_elapsed:.2f}s]")

        return self._tables

    def _generate_clif3_tables(self, course_hosps, patients, _log) -> dict[str, pd.DataFrame]:
        """Tables that exist only in CLIF 3.0, built from the course and the 2.1 tables.

        Returns 3.0 tables plus ``_output_extra`` (drain and dialysis output
        rows the converter appends to ``output``).
        """
        import time

        _log("")
        _log("  Generating CLIF 3.0 tables...")
        tb = self._tables
        out: dict[str, pd.DataFrame] = {}

        def run(name, fn):
            t = time.time()
            res = fn()
            df = res[0] if isinstance(res, tuple) else res
            out[name] = df
            _log(name, t, len(df))
            return res

        _, drain_out = run("drain", lambda: self.drain_gen.generate(
            course_hosps, tb.get("adt"), tb.get("patient_procedures")))
        _, ihd_out = run("intermittent_dialysis", lambda: self.ihd_gen.generate(course_hosps))
        out["_output_extra"] = pd.concat([d for d in (drain_out, ihd_out) if len(d)], ignore_index=True) \
            if len(drain_out) or len(ihd_out) else pd.DataFrame()
        run("patient_attributes", lambda: self.attributes_gen.generate(patients, course_hosps))
        run("radiology", lambda: self.radiology_gen.generate(course_hosps, tb.get("adt"), out["drain"]))
        run("clinical_notes_facts", lambda: self.notes_gen.generate(
            course_hosps, tb.get("adt"), tb.get("provider"), tb.get("key_icu_orders"),
            tb.get("patient_procedures")))
        run("validated_diagnosis", lambda: self.validated_dx_gen.generate(course_hosps))
        from synthetic_clif.generators.v3 import model_registry

        run("model_registry", model_registry)
        run("scores", lambda: self.scores_gen.generate(course_hosps))
        return out

    def _assign_outcomes(self, patients, hospitalizations):
        """Decide deaths and discharge destinations from each stay's course.

        Only a patient's last stay can end in death. In-hospital mortality
        rises with peak illness, organ support and age; survivors' discharge
        destinations and deaths in the following year follow the same course.
        """
        import numpy as np

        from synthetic_clif.physiology import course_for_row, course_seed_for
        from synthetic_clif.physiology.outcomes import (
            days_to_death_after_discharge,
            death_probability,
            discharge_disposition,
            outcome_rng,
        )

        hosp = hospitalizations.copy()
        pats = patients.copy()
        if len(hosp) == 0:
            return pats, hosp
        sex = pats.set_index("patient_id")["sex_category"]
        sex = sex[~sex.index.duplicated()]
        order = hosp.sort_values(["patient_id", "admission_dttm"])
        last = set(order.groupby("patient_id").tail(1).index)

        categories, deaths = [], {}
        for idx, row in hosp.iterrows():
            seed = course_seed_for(self.seed, row["hospitalization_id"])
            rec = dict(row)
            rec.update(_course_seed=seed, _sex=sex.get(row["patient_id"]), discharge_category="",
                       _patient_seed=course_seed_for(self.seed, f"patient:{row['patient_id']}"))
            course = course_for_row(rec)
            rng = outcome_rng(seed)
            p = death_probability(course)
            if idx in last and rng.random() < p:
                categories.append("Expired")
                deaths[row["patient_id"]] = row["discharge_dttm"]
                continue
            cat = discharge_disposition(course, rng, p)
            categories.append(cat)
            if idx in last:
                days = days_to_death_after_discharge(course, rng, cat, p)
                if days is not None:
                    deaths[row["patient_id"]] = row["discharge_dttm"] + pd.Timedelta(days=days)

        named = hosp["discharge_name"].notna()
        hosp["discharge_category"] = categories
        hosp["discharge_name"] = np.where(
            named, hosp["discharge_category"].str.lower().str.replace(" ", "_"), None
        )
        pats["death_dttm"] = pd.to_datetime(pats["patient_id"].map(deaths), utc=True)
        return pats, hosp

    def _course_hospitalizations(self, hospitalizations, patients) -> pd.DataFrame:
        """Hospitalizations plus the private columns the course needs."""
        from synthetic_clif.physiology import course_seed_for

        df = hospitalizations.copy()
        df["_course_seed"] = [
            course_seed_for(self.seed, h) for h in df["hospitalization_id"]
        ]
        if "patient_id" in df.columns:
            df["_patient_seed"] = [course_seed_for(self.seed, f"patient:{p}") for p in df["patient_id"]]
        if "sex_category" in patients.columns and "patient_id" in df.columns:
            sex = patients.set_index("patient_id")["sex_category"]
            sex = sex[~sex.index.duplicated()]
            df["_sex"] = df["patient_id"].map(sex)
        return df

    def _generate_big3_parallel(self, hospitalizations, adt, _log):
        """Generate vitals, labs, and respiratory in parallel.

        Uses ThreadPoolExecutor to run the 3 generators concurrently,
        with each generator using ProcessPoolExecutor internally to
        split hospitalizations into chunks across worker processes.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        from synthetic_clif.utils.parallel import parallel_generate_chunked

        _log(f"  [parallel mode: {self.workers} workers]")

        t = time.time()

        def gen_vitals():
            return parallel_generate_chunked(
                VitalsGenerator,
                hospitalizations,
                self.workers,
                self.vitals_gen._seed,
                self.mcide_dir,
                extra_dfs={"adt_df": adt},
            )

        def gen_labs():
            return parallel_generate_chunked(
                LabsGenerator,
                hospitalizations,
                self.workers,
                self.labs_gen._seed,
                self.mcide_dir,
            )

        def gen_respiratory():
            return parallel_generate_chunked(
                RespiratoryGenerator,
                hospitalizations,
                self.workers,
                self.respiratory_gen._seed,
                self.mcide_dir,
            )

        with ThreadPoolExecutor(max_workers=3) as thread_pool:
            vitals_future = thread_pool.submit(gen_vitals)
            labs_future = thread_pool.submit(gen_labs)
            respiratory_future = thread_pool.submit(gen_respiratory)

            self._tables["vitals"] = vitals_future.result()
            self._tables["labs"] = labs_future.result()
            self._tables["respiratory_support"] = respiratory_future.result()

        _log("vitals (parallel)", t, len(self._tables["vitals"]))
        _log("labs (parallel)", t, len(self._tables["labs"]))
        _log("respiratory_support (parallel)", t, len(self._tables["respiratory_support"]))

    def to_parquet(self, output_dir: Path, prefix: str = "clif_") -> None:
        """Write each table to a parquet file.

        Args:
            output_dir: Directory to write parquet files to
            prefix: Filename prefix (default "clif_" for CLIF compatibility)
        """
        if not self._tables:
            self.generate()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing parquet files to {output_dir}...")

        for table_name, df in self._tables.items():
            if len(df) == 0:
                print(f"  Skipping empty table: {table_name}")
                continue

            output_path = output_dir / f"{prefix}{table_name}.parquet"
            df.to_parquet(output_path, index=False)
            print(f"  Wrote {prefix}{table_name}.parquet ({len(df)} rows)")

        print("Done!")

    def to_csv(self, output_dir: Path, prefix: str = "clif_") -> None:
        """Write each table to a CSV file.

        Args:
            output_dir: Directory to write CSV files to
            prefix: Filename prefix (default "clif_" for CLIF compatibility)
        """
        if not self._tables:
            self.generate()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Writing CSV files to {output_dir}...")

        for table_name, df in self._tables.items():
            if len(df) == 0:
                print(f"  Skipping empty table: {table_name}")
                continue

            output_path = output_dir / f"{prefix}{table_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"  Wrote {prefix}{table_name}.csv ({len(df)} rows)")

        print("Done!")

    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Get a specific table by name.

        Args:
            table_name: Name of the table to retrieve

        Returns:
            DataFrame or None if not generated yet
        """
        return self._tables.get(table_name)

    def summary(self) -> pd.DataFrame:
        """Get a summary of all generated tables.

        Returns:
            DataFrame with table names and row counts
        """
        if not self._tables:
            return pd.DataFrame(columns=["table", "rows"])

        data = [
            {"table": name, "rows": len(df)} for name, df in self._tables.items()
        ]
        return pd.DataFrame(data).sort_values("table")
