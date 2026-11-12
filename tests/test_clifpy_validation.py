"""CLIFPy validation tests for synthetic CLIF data.

Generates a small synthetic dataset and validates it using the CLIFPy
package to ensure schema compliance with the CLIF 2.1.0 specification
(columns, data types, vocabularies). clifpy reports units, value ranges,
keys and timestamps only as data-quality warnings, so those are enforced in
tests/test_integrity.py instead.
"""

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from synthetic_clif.generators.dataset import SyntheticCLIFDataset
from tests.clifpy_errors import DTYPE_MISMATCH, of_type, structural_errors

# Tables that CLIFPy can validate (18 total)
CLIFPY_VALIDATED_TABLES = [
    "patient",
    "hospitalization",
    "adt",
    "labs",
    "vitals",
    "medication_admin_continuous",
    "medication_admin_intermittent",
    "patient_assessments",
    "respiratory_support",
    "position",
    "hospital_diagnosis",
    "microbiology_culture",
    "crrt_therapy",
    "patient_procedures",
    "microbiology_susceptibility",
    "ecmo_mcs",
    "microbiology_nonculture",
    "code_status",
]


@pytest.fixture(scope="module")
def clifpy_results():
    """Generate a small dataset, write to parquet, and run CLIFPy validation.

    Returns dict mapping table_name -> (is_valid, structural_errors, all_errors)
    for tables that were actually generated (non-empty).
    """
    clifpy = pytest.importorskip("clifpy", reason="clifpy not installed")
    ClifOrchestrator = clifpy.ClifOrchestrator

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Generate dataset large enough for rare tables (ECMO, CRRT) to appear
        dataset = SyntheticCLIFDataset(
            n_patients=100,
            n_hospitalizations=120,
            seed=42,
            include_concept_tables=True,
            clif_version="2.1",  # validated against clifpy's 2.1 schema;
            # 3.0 output is checked in test_clif3.py
        )
        dataset.generate()
        dataset.to_parquet(output_dir)

        # Discover which tables were actually written
        written_tables = [
            t for t in CLIFPY_VALIDATED_TABLES
            if (output_dir / f"clif_{t}.parquet").exists()
        ]

        # Initialize CLIFPy orchestrator
        co = ClifOrchestrator(
            data_directory=str(output_dir),
            filetype="parquet",
            timezone="UTC",
            output_directory=str(output_dir / "clifpy_output"),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            co.initialize(tables=written_tables)
            co.validate_all()

        # Collect results
        results = {}
        for table_name in written_tables:
            table_obj = getattr(co, table_name, None)
            if table_obj is not None:
                all_errors = list(table_obj.errors)
                structural = structural_errors(all_errors, columns=table_obj.df.columns)
                results[table_name] = (
                    table_obj.isvalid(),
                    structural,
                    all_errors,
                )

        return results


class TestCLIFPyValidation:
    """Validate synthetic CLIF data against CLIFPy schema checks."""

    def test_tables_loaded(self, clifpy_results):
        """Test that CLIFPy loaded a meaningful number of tables."""
        assert len(clifpy_results) >= 16, (
            f"Only {len(clifpy_results)} tables loaded, expected >= 16"
        )

    def test_no_missing_columns(self, clifpy_results):
        """Test that no table has missing required columns."""
        for table_name, (_, structural, _) in clifpy_results.items():
            missing = of_type(structural, "Missing Columns", "Missing Required Columns")
            assert not missing, (
                f"Table '{table_name}' has missing columns: {missing}"
            )

    def test_no_datatype_mismatches(self, clifpy_results):
        """Test that no table has non-castable datatype mismatches."""
        for table_name, (_, structural, _) in clifpy_results.items():
            mismatches = of_type(structural, DTYPE_MISMATCH)
            assert not mismatches, (
                f"Table '{table_name}' has type mismatches: {mismatches}"
            )

    def test_no_invalid_categories(self, clifpy_results):
        """Test that no table has invalid categorical values."""
        for table_name, (_, structural, _) in clifpy_results.items():
            invalid = of_type(structural, "Invalid Categorical Values")
            assert not invalid, (
                f"Table '{table_name}' has invalid categories: {invalid}"
            )

    def test_no_invalid_units(self, clifpy_results):
        """Test that no table has invalid measurement units."""
        for table_name, (_, structural, _) in clifpy_results.items():
            bad_units = of_type(structural, "Invalid Units")
            assert not bad_units, (
                f"Table '{table_name}' has invalid units: {bad_units}"
            )

    def test_no_structural_errors(self, clifpy_results):
        """Test that no table has any structural validation errors."""
        failures = {}
        for table_name, (_, structural, _) in clifpy_results.items():
            if structural:
                failures[table_name] = structural

        assert not failures, (
            f"Structural validation errors found: {failures}"
        )

    def test_some_tables_fully_valid(self, clifpy_results):
        """Test that at least some tables pass full CLIFPy validation.

        CLIFPy's isvalid() also flags data quality warnings (high missingness,
        distribution shifts, coverage gaps) that are expected in synthetic data
        with intentional artifacts. Mark as xfail until data quality warnings
        are tuned.
        """
        fully_valid = [
            name for name, (is_valid, _, _) in clifpy_results.items()
            if is_valid
        ]
        if len(fully_valid) == 0:
            pytest.xfail(
                "No tables fully valid — expected due to intentional "
                "missingness and data quality warnings in synthetic data"
            )


# ---------------------------------------------------------------------------
# The filter itself (issue #53): inject known schema errors and make sure the
# tests above would see them. If clifpy renames its error types again, these
# fail instead of every structural test passing silently.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def v21_tables():
    ds = SyntheticCLIFDataset(n_patients=20, n_hospitalizations=24, seed=3, clif_version="2.1")
    return ds.generate(verbose=False)


def _clifpy_errors(table_name: str, df: pd.DataFrame) -> list[dict]:
    clifpy_tables = pytest.importorskip("clifpy.tables", reason="clifpy not installed")
    cls = {n.lower(): getattr(clifpy_tables, n) for n in dir(clifpy_tables)
           if isinstance(getattr(clifpy_tables, n), type)}[table_name.replace("_", "")]
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            obj = cls(data_directory=tmp, filetype="parquet", timezone="UTC", data=df,
                      clif_version="2.1", output_directory=tmp)
        except TypeError:  # clifpy releases without these arguments
            obj = cls(data_directory=tmp, filetype="parquet", timezone="UTC", data=df)
        obj.validate()
        return list(obj.errors)


class TestFilterCatchesErrors:
    def test_clean_table_has_no_structural_errors(self, v21_tables):
        df = v21_tables["vitals"]
        assert structural_errors(_clifpy_errors("vitals", df), df.columns) == []

    def test_missing_required_column(self, v21_tables):
        df = v21_tables["vitals"].drop(columns=["vital_category"])
        found = structural_errors(_clifpy_errors("vitals", df), df.columns)
        assert of_type(found, "Missing Columns", "Missing Required Columns"), found

    def test_invalid_category(self, v21_tables):
        df = v21_tables["vitals"].copy()
        df.loc[df.index[:5], "vital_category"] = "not_a_vital"
        found = structural_errors(_clifpy_errors("vitals", df), df.columns)
        assert of_type(found, "Invalid Categorical Values"), found

    def test_wrong_dtype(self, v21_tables):
        df = v21_tables["vitals"].copy()
        df["vital_value"] = "abc"
        found = structural_errors(_clifpy_errors("vitals", df), df.columns)
        assert of_type(found, DTYPE_MISMATCH), found
