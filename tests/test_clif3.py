"""Tests for CLIF 3.0.0 output (``clif_version="3.0"``).

3.0.0 is the default output. The 3.0 tables are built by converting the 2.1
tables the generators emit (``synthetic_clif.clif3``), so these tests check
three things: the vocabulary mapping, conformance with the vendored 3.0 schema,
and that 2.1 output is still available and unchanged.
"""

import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import pytest

from synthetic_clif.clif3 import convert_to_clif3, normalize_category_value, schema
from synthetic_clif.clif3.validate import validate_table, validate_tables
from synthetic_clif.clif3.vocab import to_v3_value
from synthetic_clif.generators.dataset import SyntheticCLIFDataset, normalize_clif_version
from tests.clifpy_errors import structural_errors

V21_ONLY_TABLES = {"ecmo_mcs", "intake_output", "key_icu_orders"}



@pytest.fixture(scope="module")
def v3_dataset():
    ds = SyntheticCLIFDataset(n_patients=40, n_hospitalizations=50, seed=7, clif_version="3.0")
    ds.generate(verbose=False)
    return ds


@pytest.fixture(scope="module")
def v3_tables(v3_dataset):
    return v3_dataset._tables


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
class TestVocabulary:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Assist Control-Volume Control", "assist_control_volume_control"),
            ("Pressure Support/CPAP", "pressure_support_or_cpap"),
            ("Hospice & Home", "hospice_and_home"),
            ("  Room Air ", "room_air"),
            ("ICU", "icu"),
            (None, None),
        ],
    )
    def test_normalize_category_value(self, raw, expected):
        assert normalize_category_value(raw) == expected

    def test_official_renames(self):
        pv = schema.permissible_values("respiratory_support", "device_category")
        assert to_v3_value("respiratory_support", "device_category", "High Flow NC", pv) == "hfnc"
        pv = schema.permissible_values("respiratory_support", "mode_category")
        assert (
            to_v3_value("respiratory_support", "mode_category", "Pressure Support/CPAP", pv)
            == "ps_or_cpap"
        )
        pv = schema.permissible_values("hospitalization", "discharge_category")
        assert (
            to_v3_value("hospitalization", "discharge_category", "Long Term Care Hospital (LTACH)", pv)
            == "ltach"
        )

    def test_valid_value_unchanged(self):
        pv = schema.permissible_values("labs", "lab_category")
        assert to_v3_value("labs", "lab_category", "sodium", pv) == "sodium"

    @pytest.mark.parametrize(
        "raw, expected",
        [("2.1", "2.1"), ("2.1.0", "2.1"), ("3", "3.0"), ("3.0.0", "3.0"), ("v3.0", "3.0")],
    )
    def test_normalize_clif_version(self, raw, expected):
        assert normalize_clif_version(raw) == expected

    def test_unknown_version_rejected(self):
        with pytest.raises(ValueError):
            SyntheticCLIFDataset(n_patients=2, n_hospitalizations=2, clif_version="2.0")


# --------------------------------------------------------------------------
# Schema conformance
# --------------------------------------------------------------------------
class TestSchemaConformance:
    def test_builtin_validator_passes(self, v3_tables):
        errors = {t: e for t, e in validate_tables(v3_tables).items() if e}
        assert not errors, f"CLIF 3.0 validation errors: {errors}"

    def test_table_names_are_3_0(self, v3_tables):
        names = set(v3_tables)
        assert not names & V21_ONLY_TABLES
        assert names <= set(schema.table_names())
        for t in ("patient", "hospitalization", "adt", "vitals", "labs", "input", "output",
                  "consult_orders", "line", "airway"):
            assert t in names

    def test_columns_in_schema_order(self, v3_tables):
        for name, df in v3_tables.items():
            assert list(df.columns) == schema.columns(name), name

    def test_datetimes_are_utc(self, v3_tables):
        for name, df in v3_tables.items():
            for c in schema.table(name)["columns"]:
                if c["data_type"].upper() == "DATETIME":
                    assert str(df[c["name"]].dtype) == "datetime64[ns, UTC]", (name, c["name"])

    def test_validator_catches_bad_values(self, v3_tables):
        df = v3_tables["adt"].copy()
        df.loc[df.index[0], "location_category"] = "ICU"  # 2.1 spelling
        df = df.drop(columns=["room_id"])
        types = {e["type"] for e in validate_table("adt", df)}
        assert {"invalid_categorical", "missing_columns"} <= types

    def test_new_3_0_columns_filled(self, v3_tables):
        assert v3_tables["adt"]["room_id"].notna().all()
        assert v3_tables["vitals"]["measured_dttm"].notna().all()
        mai = v3_tables["medication_admin_intermittent"]
        assert mai["administering_provider_id"].notna().all()
        providers = set(v3_tables["provider"]["provider_id"].astype(str))
        assert mai["administering_provider_id"].astype(str).isin(providers).all()

    def test_troponin_in_ng_per_l(self, v3_tables):
        labs = v3_tables["labs"]
        trop = labs[labs["lab_category"].isin(["troponin_i", "troponin_t"])]
        if len(trop):
            assert set(trop["reference_unit"].dropna().str.lower()) <= {"ng/l"}

    def test_conversion_is_deterministic(self, v3_dataset):
        again = convert_to_clif3(v3_dataset._tables_v21, seed=v3_dataset.seed,
                                 native_tables=v3_dataset._tables_v3_native)
        for name, df in v3_dataset._tables.items():
            pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                          again[name].reset_index(drop=True), obj=name)


# --------------------------------------------------------------------------
# Derived 3.0 tables
# --------------------------------------------------------------------------
class TestDerivedTables:
    def test_airway_matches_imv(self, v3_tables):
        rs = v3_tables["respiratory_support"]
        imv_hosps = set(rs.loc[rs["device_category"] == "imv", "hospitalization_id"])
        airway = v3_tables["airway"]
        assert set(airway["hospitalization_id"]) <= imv_hosps
        ett = airway[airway["airway_category"] == "ett"]
        done = ett.dropna(subset=["removed_dttm"])
        assert (done["removed_dttm"] >= done["placed_dttm"]).all()

    def test_lines_within_stay(self, v3_tables):
        line = v3_tables["line"].merge(
            v3_tables["hospitalization"][["hospitalization_id", "admission_dttm"]],
            on="hospitalization_id",
        )
        assert (line["placed_dttm"] >= line["admission_dttm"]).all()
        removed = line.dropna(subset=["removed_dttm"])
        assert (removed["removed_dttm"] >= removed["placed_dttm"]).all()

    def test_output_has_groups(self, v3_tables):
        out = v3_tables["output"]
        assert len(out)
        assert out["output_group"].notna().all()


# --------------------------------------------------------------------------
# 3.0 default, 2.1 still available
# --------------------------------------------------------------------------
class TestVersionSelection:
    def test_default_is_3_0(self):
        ds = SyntheticCLIFDataset(n_patients=3, n_hospitalizations=4, seed=7)
        assert ds.clif_version == "3.0"
        tables = ds.generate(verbose=False)
        assert "input" in tables
        assert not set(tables) & V21_ONLY_TABLES

    def test_2_1_still_available(self):
        ds = SyntheticCLIFDataset(n_patients=3, n_hospitalizations=4, seed=7, clif_version="2.1")
        tables = ds.generate(verbose=False)
        assert "intake_output" in tables
        assert "input" not in tables and "airway" not in tables

    def test_same_seed_same_ids(self, v3_dataset):
        v21 = v3_dataset._tables_v21
        v3 = v3_dataset._tables
        assert list(v21["patient"]["patient_id"]) == list(v3["patient"]["patient_id"])
        assert list(v21["hospitalization"]["hospitalization_id"]) == list(
            v3["hospitalization"]["hospitalization_id"]
        )

    def test_2_1_matches_unconverted(self, v3_dataset):
        ds = SyntheticCLIFDataset(n_patients=40, n_hospitalizations=50, seed=7, clif_version="2.1")
        v21 = ds.generate(verbose=False)
        assert set(v21) == set(v3_dataset._tables_v21)
        pd.testing.assert_frame_equal(v21["vitals"], v3_dataset._tables_v21["vitals"])


# --------------------------------------------------------------------------
# CLI and clifpy
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flag, present, absent",
    [([], "input", "intake_output"), (["--clif-version", "2.1"], "intake_output", "input")],
)
def test_cli_clif_version(tmp_path, flag, present, absent):
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, "-m", "synthetic_clif", "--patients", "3", "--hospitalizations", "4",
         "--output", str(out), *flag],
        check=True, capture_output=True,
    )
    assert (out / f"clif_{present}.parquet").exists()
    assert not (out / f"clif_{absent}.parquet").exists()


def test_clifpy_3_0_no_structural_errors(v3_dataset):
    clifpy_tables = pytest.importorskip("clifpy.tables", reason="clifpy not installed")
    classes = {
        n.lower(): getattr(clifpy_tables, n)
        for n in dir(clifpy_tables)
        if isinstance(getattr(clifpy_tables, n), type)
    }
    failures = {}
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v3_dataset.to_parquet(Path(tmp))
        checked = 0
        for name in v3_dataset._tables:
            cls = classes.get(name.replace("_", ""))
            path = Path(tmp) / f"clif_{name}.parquet"
            if cls is None or not path.exists():
                continue
            df = pd.read_parquet(path)
            try:
                obj = cls(data_directory=tmp, filetype="parquet", timezone="UTC", data=df,
                          clif_version="3.0", output_directory=tmp)
            except TypeError:
                pytest.skip("installed clifpy has no clif_version argument")
            obj.validate()
            checked += 1
            bad = structural_errors(obj.errors, df.columns)
            if bad:
                failures[name] = bad
    assert checked >= 10
    assert not failures, failures
