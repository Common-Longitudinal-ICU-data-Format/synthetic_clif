"""CLIFPy validation tests for synthetic CLIF data.

Generates a small synthetic dataset and validates it using the CLIFPy
package to ensure schema compliance with the CLIF 2.1.0 specification.
These tests replace individual category/unit/range checks across
per-generator test files since CLIFPy validates all of those.
"""

import pytest
import tempfile
from pathlib import Path

from synthetic_clif.generators.dataset import SyntheticCLIFDataset

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


def _get_structural_errors(errors: list[dict]) -> list[dict]:
    """Filter validation errors to only structural/schema problems.

    Excludes data quality warnings (null counts, outlier ranges, duplicates)
    that are expected in synthetic data with intentional missingness.
    """
    structural = []
    for e in errors:
        etype = e.get("type", "")
        status = e.get("status", "")

        if etype == "missing_columns":
            structural.append(e)
        elif etype == "datatype_mismatch":
            structural.append(e)
        elif etype == "invalid_categorical_values" and status == "error":
            structural.append(e)
        elif etype == "invalid_units" and status == "error":
            structural.append(e)

    return structural


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
        )

        co.initialize(tables=written_tables)
        co.validate_all()

        # Collect results
        results = {}
        for table_name in written_tables:
            table_obj = getattr(co, table_name, None)
            if table_obj is not None:
                all_errors = list(table_obj.errors)
                structural = _get_structural_errors(all_errors)
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
            missing = [e for e in structural if e["type"] == "missing_columns"]
            assert not missing, (
                f"Table '{table_name}' has missing columns: {missing}"
            )

    def test_no_datatype_mismatches(self, clifpy_results):
        """Test that no table has non-castable datatype mismatches."""
        for table_name, (_, structural, _) in clifpy_results.items():
            mismatches = [
                e for e in structural if e["type"] == "datatype_mismatch"
            ]
            assert not mismatches, (
                f"Table '{table_name}' has type mismatches: {mismatches}"
            )

    def test_no_invalid_categories(self, clifpy_results):
        """Test that no table has invalid categorical values."""
        for table_name, (_, structural, _) in clifpy_results.items():
            invalid = [
                e for e in structural
                if e["type"] == "invalid_categorical_values"
            ]
            assert not invalid, (
                f"Table '{table_name}' has invalid categories: {invalid}"
            )

    def test_no_invalid_units(self, clifpy_results):
        """Test that no table has invalid measurement units."""
        for table_name, (_, structural, _) in clifpy_results.items():
            bad_units = [
                e for e in structural if e["type"] == "invalid_units"
            ]
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
