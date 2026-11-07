"""Tests for parallel dataset generation."""

import pytest
import pandas as pd

from synthetic_clif.generators.dataset import SyntheticCLIFDataset


# Use small dataset sizes to keep tests fast
N_PATIENTS = 5
N_HOSPITALIZATIONS = 8
SEED = 42
WORKERS = 2


@pytest.fixture
def parallel_dataset():
    """Generate a dataset using parallel mode."""
    dataset = SyntheticCLIFDataset(
        n_patients=N_PATIENTS,
        n_hospitalizations=N_HOSPITALIZATIONS,
        seed=SEED,
        include_concept_tables=True,
        workers=WORKERS,
        clif_version="2.1",  # table lists below are CLIF 2.1 names
    )
    return dataset.generate(verbose=False)


@pytest.fixture
def sequential_dataset():
    """Generate a dataset using sequential mode."""
    dataset = SyntheticCLIFDataset(
        n_patients=N_PATIENTS,
        n_hospitalizations=N_HOSPITALIZATIONS,
        seed=SEED,
        include_concept_tables=True,
        workers=1,
        clif_version="2.1",  # compared against the 2.1 parallel run
    )
    return dataset.generate(verbose=False)


EXPECTED_BETA_TABLES = {
    "patient",
    "hospitalization",
    "adt",
    "vitals",
    "labs",
    "respiratory_support",
    "medication_admin_continuous",
    "medication_admin_intermittent",
    "microbiology_culture",
    "microbiology_susceptibility",
    "patient_assessments",
    "patient_procedures",
    "hospital_diagnosis",
    "code_status",
    "position",
    "crrt_therapy",
}

EXPECTED_CONCEPT_TABLES = {
    "clinical_trial",
    "ecmo_mcs",
    "intake_output",
    "invasive_hemodynamics",
    "key_icu_orders",
    "medication_orders",
    "microbiology_nonculture",
    "patient_diagnosis",
    "place_based_index",
    "provider",
    "therapy_details",
    "transfusion",
}


class TestParallelProducesSameTables:
    """Parallel mode should produce the same set of table names as sequential."""

    def test_parallel_produces_same_tables(self, parallel_dataset, sequential_dataset):
        assert set(parallel_dataset.keys()) == set(sequential_dataset.keys())

    def test_parallel_has_all_beta_tables(self, parallel_dataset):
        assert EXPECTED_BETA_TABLES.issubset(set(parallel_dataset.keys()))

    def test_parallel_has_all_concept_tables(self, parallel_dataset):
        assert EXPECTED_CONCEPT_TABLES.issubset(set(parallel_dataset.keys()))


class TestParallelRowCountsReasonable:
    """Parallel mode should produce non-trivial output for each table."""

    def test_patient_count(self, parallel_dataset):
        assert len(parallel_dataset["patient"]) == N_PATIENTS

    def test_hospitalization_count(self, parallel_dataset):
        assert len(parallel_dataset["hospitalization"]) == N_HOSPITALIZATIONS

    def test_vitals_nontrivial(self, parallel_dataset):
        assert len(parallel_dataset["vitals"]) > N_HOSPITALIZATIONS

    def test_labs_nontrivial(self, parallel_dataset):
        assert len(parallel_dataset["labs"]) > N_HOSPITALIZATIONS

    def test_respiratory_nontrivial(self, parallel_dataset):
        assert len(parallel_dataset["respiratory_support"]) > N_HOSPITALIZATIONS

    def test_adt_nontrivial(self, parallel_dataset):
        assert len(parallel_dataset["adt"]) > 0

    def test_medications_nontrivial(self, parallel_dataset):
        assert len(parallel_dataset["medication_admin_continuous"]) > 0
        assert len(parallel_dataset["medication_admin_intermittent"]) > 0


class TestParallelReferentialIntegrity:
    """All hospitalization_ids in parallel output should be valid."""

    def test_vitals_hosp_ids(self, parallel_dataset):
        valid_ids = set(parallel_dataset["hospitalization"]["hospitalization_id"])
        vitals = parallel_dataset["vitals"]
        if len(vitals) > 0:
            assert set(vitals["hospitalization_id"]).issubset(valid_ids)

    def test_labs_hosp_ids(self, parallel_dataset):
        valid_ids = set(parallel_dataset["hospitalization"]["hospitalization_id"])
        labs = parallel_dataset["labs"]
        if len(labs) > 0:
            assert set(labs["hospitalization_id"]).issubset(valid_ids)

    def test_respiratory_hosp_ids(self, parallel_dataset):
        valid_ids = set(parallel_dataset["hospitalization"]["hospitalization_id"])
        resp = parallel_dataset["respiratory_support"]
        if len(resp) > 0:
            assert set(resp["hospitalization_id"]).issubset(valid_ids)

    def test_all_timeseries_hosp_ids(self, parallel_dataset):
        """Check referential integrity across all time-series tables."""
        valid_ids = set(parallel_dataset["hospitalization"]["hospitalization_id"])
        for table_name in [
            "vitals", "labs", "adt", "respiratory_support",
            "medication_admin_continuous", "medication_admin_intermittent",
            "patient_assessments", "patient_procedures",
        ]:
            df = parallel_dataset[table_name]
            if len(df) > 0 and "hospitalization_id" in df.columns:
                invalid = set(df["hospitalization_id"]) - valid_ids
                assert not invalid, f"Invalid hosp IDs in {table_name}: {invalid}"


class TestParallelReproducibility:
    """Same (seed, workers) should produce the same output across runs."""

    def test_parallel_reproducibility(self):
        dataset1 = SyntheticCLIFDataset(
            n_patients=N_PATIENTS,
            n_hospitalizations=N_HOSPITALIZATIONS,
            seed=SEED,
            workers=WORKERS,
            include_concept_tables=False,
        )
        tables1 = dataset1.generate(verbose=False)

        dataset2 = SyntheticCLIFDataset(
            n_patients=N_PATIENTS,
            n_hospitalizations=N_HOSPITALIZATIONS,
            seed=SEED,
            workers=WORKERS,
            include_concept_tables=False,
        )
        tables2 = dataset2.generate(verbose=False)

        # Vitals should be identical
        pd.testing.assert_frame_equal(
            tables1["vitals"].sort_values(
                ["hospitalization_id", "recorded_dttm", "vital_category"]
            ).reset_index(drop=True),
            tables2["vitals"].sort_values(
                ["hospitalization_id", "recorded_dttm", "vital_category"]
            ).reset_index(drop=True),
        )

        # Labs should be identical
        pd.testing.assert_frame_equal(
            tables1["labs"].sort_values(
                ["hospitalization_id", "lab_result_dttm", "lab_category"]
            ).reset_index(drop=True),
            tables2["labs"].sort_values(
                ["hospitalization_id", "lab_result_dttm", "lab_category"]
            ).reset_index(drop=True),
        )

        # Respiratory should be identical
        pd.testing.assert_frame_equal(
            tables1["respiratory_support"].sort_values(
                ["hospitalization_id", "recorded_dttm"]
            ).reset_index(drop=True),
            tables2["respiratory_support"].sort_values(
                ["hospitalization_id", "recorded_dttm"]
            ).reset_index(drop=True),
        )


class TestSequentialModeUnchanged:
    """workers=1 should use the original sequential code path."""

    def test_sequential_mode_unchanged(self):
        """Sequential output should be identical across runs with same seed."""
        dataset1 = SyntheticCLIFDataset(
            n_patients=N_PATIENTS,
            n_hospitalizations=N_HOSPITALIZATIONS,
            seed=SEED,
            workers=1,
            include_concept_tables=False,
        )
        tables1 = dataset1.generate(verbose=False)

        dataset2 = SyntheticCLIFDataset(
            n_patients=N_PATIENTS,
            n_hospitalizations=N_HOSPITALIZATIONS,
            seed=SEED,
            workers=1,
            include_concept_tables=False,
        )
        tables2 = dataset2.generate(verbose=False)

        # Check patient table non-timestamp columns match
        non_dt_cols = [c for c in tables1["patient"].columns if c != "death_dttm"]
        pd.testing.assert_frame_equal(
            tables1["patient"][non_dt_cols], tables2["patient"][non_dt_cols]
        )

        # Check vitals match exactly (same seed, same order)
        pd.testing.assert_frame_equal(tables1["vitals"], tables2["vitals"])

    def test_workers_1_default(self):
        """Default workers=1 should be set correctly."""
        dataset = SyntheticCLIFDataset(
            n_patients=N_PATIENTS,
            n_hospitalizations=N_HOSPITALIZATIONS,
            seed=SEED,
        )
        assert dataset.workers == 1


class TestParallelClif3:
    """Parallel generation also yields valid CLIF 3.0.0 output (the default)."""

    def test_parallel_3_0_matches_sequential_and_validates(self):
        from synthetic_clif.clif3.validate import validate_tables

        kwargs = dict(n_patients=N_PATIENTS, n_hospitalizations=N_HOSPITALIZATIONS,
                      seed=SEED, include_concept_tables=True)
        par = SyntheticCLIFDataset(workers=WORKERS, **kwargs).generate(verbose=False)
        seq = SyntheticCLIFDataset(workers=1, **kwargs).generate(verbose=False)
        assert set(par) == set(seq)
        errors = {t: e for t, e in validate_tables(par).items() if e}
        assert not errors, errors
