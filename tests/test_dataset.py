"""Integration tests for dataset orchestrator."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from synthetic_clif.generators.dataset import SyntheticCLIFDataset


class TestSyntheticCLIFDataset:
    """Integration tests for SyntheticCLIFDataset."""

    def test_generate_basic(self):
        """Test basic dataset generation."""
        dataset = SyntheticCLIFDataset(
            n_patients=5, n_hospitalizations=8, seed=42
        )
        tables = dataset.generate()

        assert "patient" in tables
        assert "hospitalization" in tables
        assert "adt" in tables
        assert "vitals" in tables
        assert "labs" in tables
        assert "respiratory_support" in tables
        assert "medication_admin_continuous" in tables
        assert "medication_admin_intermittent" in tables
        assert "microbiology_culture" in tables
        assert "microbiology_susceptibility" in tables
        assert "patient_assessments" in tables
        assert "patient_procedures" in tables
        assert "hospital_diagnosis" in tables
        assert "code_status" in tables
        assert "position" in tables
        assert "crrt_therapy" in tables

    def test_generate_with_concept_tables(self):
        """Test generation includes concept tables."""
        dataset = SyntheticCLIFDataset(
            n_patients=5,
            n_hospitalizations=8,
            seed=42,
            include_concept_tables=True,
            clif_version="2.1",  # checks the 2.1 concept table names
        )
        tables = dataset.generate()

        assert "clinical_trial" in tables
        assert "ecmo_mcs" in tables
        assert "intake_output" in tables
        assert "invasive_hemodynamics" in tables
        assert "key_icu_orders" in tables
        assert "medication_orders" in tables
        assert "microbiology_nonculture" in tables
        assert "patient_diagnosis" in tables
        assert "place_based_index" in tables
        assert "provider" in tables
        assert "therapy_details" in tables
        assert "transfusion" in tables

    def test_generate_without_concept_tables(self):
        """Test generation excludes concept tables when requested."""
        dataset = SyntheticCLIFDataset(
            n_patients=5,
            n_hospitalizations=8,
            seed=42,
            include_concept_tables=False,
        )
        tables = dataset.generate()

        assert "clinical_trial" not in tables
        assert "ecmo_mcs" not in tables

    def test_referential_integrity_patient(self, small_dataset):
        """Test that all patient IDs in hospitalization exist in patient table."""
        patients = small_dataset["patient"]
        hospitalizations = small_dataset["hospitalization"]

        valid_patient_ids = set(patients["patient_id"])
        for pid in hospitalizations["patient_id"]:
            assert pid in valid_patient_ids

    def test_referential_integrity_hospitalization(self, small_dataset):
        """Test that all hospitalization IDs in time-series tables exist."""
        hospitalizations = small_dataset["hospitalization"]
        valid_hosp_ids = set(hospitalizations["hospitalization_id"])

        # Check several time-series tables
        for table_name in ["vitals", "labs", "adt", "respiratory_support"]:
            df = small_dataset[table_name]
            if len(df) > 0:
                for hosp_id in df["hospitalization_id"]:
                    assert hosp_id in valid_hosp_ids, f"Invalid hosp_id in {table_name}"

    def test_referential_integrity_microbiology(self, small_dataset):
        """Test microbiology susceptibility references valid organisms."""
        cultures = small_dataset["microbiology_culture"]
        susceptibilities = small_dataset["microbiology_susceptibility"]

        if len(susceptibilities) > 0:
            valid_organism_ids = set(cultures["organism_id"].dropna())
            for oid in susceptibilities["organism_id"]:
                assert oid in valid_organism_ids

    def test_row_counts(self, small_dataset):
        """Test that tables have expected row counts."""
        # Patient count
        assert len(small_dataset["patient"]) == 5

        # Hospitalization count
        assert len(small_dataset["hospitalization"]) == 8

        # Time-series tables should have multiple rows per hospitalization
        assert len(small_dataset["vitals"]) > 8
        assert len(small_dataset["labs"]) > 8

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        dataset1 = SyntheticCLIFDataset(
            n_patients=5, n_hospitalizations=8, seed=42,
        )
        tables1 = dataset1.generate()

        dataset2 = SyntheticCLIFDataset(
            n_patients=5, n_hospitalizations=8, seed=42,
        )
        tables2 = dataset2.generate()

        # Check patient table non-timestamp columns match
        non_dt_cols = [c for c in tables1["patient"].columns if c != "death_dttm"]
        pd.testing.assert_frame_equal(
            tables1["patient"][non_dt_cols], tables2["patient"][non_dt_cols]
        )

        # Check hospitalization non-timestamp columns match
        non_dt_cols = [c for c in tables1["hospitalization"].columns
                       if "dttm" not in c]
        pd.testing.assert_frame_equal(
            tables1["hospitalization"][non_dt_cols],
            tables2["hospitalization"][non_dt_cols],
        )

    def test_summary(self):
        """Test summary method."""
        dataset = SyntheticCLIFDataset(
            n_patients=5, n_hospitalizations=8, seed=42
        )
        dataset.generate()

        summary = dataset.summary()
        assert "table" in summary.columns
        assert "rows" in summary.columns
        assert len(summary) > 0

    def test_get_table(self):
        """Test get_table method."""
        dataset = SyntheticCLIFDataset(
            n_patients=5, n_hospitalizations=8, seed=42
        )
        dataset.generate()

        patients = dataset.get_table("patient")
        assert patients is not None
        assert len(patients) == 5

        # Non-existent table
        assert dataset.get_table("nonexistent") is None

    def test_timestamp_consistency(self, small_dataset):
        """Test that timestamps are within hospitalization bounds."""
        hospitalizations = small_dataset["hospitalization"]
        vitals = small_dataset["vitals"]

        hosp_lookup = hospitalizations.set_index("hospitalization_id")

        for _, row in vitals.iterrows():
            hosp_id = row["hospitalization_id"]
            hosp = hosp_lookup.loc[hosp_id]

            assert row["recorded_dttm"] >= hosp["admission_dttm"]
            if pd.notna(hosp["discharge_dttm"]):
                assert row["recorded_dttm"] <= hosp["discharge_dttm"]
