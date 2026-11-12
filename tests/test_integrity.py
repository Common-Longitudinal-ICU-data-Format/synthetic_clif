"""Integrity checks on generated output (issue #53).

Two kinds of test:

* the generated dataset, in both CLIF versions, passes every check in
  ``synthetic_clif.validation`` (one test per check, so a failure names it);
* each check fails when an error of its kind is injected into a copy of the
  dataset ("mutation" tests), and the range rules actually apply to the
  values generated, so a check cannot pass by matching nothing.

New tables are picked up by the generic checks with no change here. A new
check gets a zero-issue test automatically (the parametrize list is the
registry) and should get a mutation test below.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from synthetic_clif.generators.dataset import SyntheticCLIFDataset
from synthetic_clif.validation import CHECKS, check_dataset, format_issues, range_coverage


@pytest.fixture(scope="module")
def dataset():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = SyntheticCLIFDataset(n_patients=80, n_hospitalizations=100, seed=11, clif_version="3.0")
        ds.generate(verbose=False)
    return ds


@pytest.fixture(scope="module")
def v3(dataset):
    return dataset._tables


@pytest.fixture(scope="module")
def v21(dataset):
    return dataset._tables_v21


def _copy(tables):
    return {k: v.copy() for k, v in tables.items()}


def _issues(tables, check, version="3.0"):
    return check_dataset(tables, version, checks=[check])


def _assert_caught(tables, check, version="3.0", table=None):
    found = _issues(tables, check, version)
    if table is not None:
        found = [i for i in found if i["table"] == table]
    assert found, f"{check} did not catch the injected error"
    return found


# --------------------------------------------------------------------------
# Generated data passes
# --------------------------------------------------------------------------
@pytest.mark.parametrize("check", sorted(CHECKS))
def test_v3_passes(v3, check):
    issues = _issues(v3, check, "3.0")
    assert not issues, format_issues(issues)


@pytest.mark.parametrize("check", sorted(set(CHECKS) - {"schema"}))
def test_v21_passes(v21, check):
    issues = _issues(v21, check, "2.1")
    assert not issues, format_issues(issues)


def test_unknown_check_is_an_error(v3):
    with pytest.raises(KeyError):
        check_dataset(v3, "3.0", checks=["no_such_check"])


# --------------------------------------------------------------------------
# Rules apply to what is generated (no silent pass by matching nothing)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("version", ["3.0", "2.1"])
@pytest.mark.parametrize("table, column", [
    ("vitals", "vital_value"),
    ("labs", "lab_value_numeric"),
    ("respiratory_support", "fio2_set"),
    ("respiratory_support", "peep_set"),
    ("patient_assessments", "numerical_value"),
    ("crrt_therapy", "blood_flow_rate"),
    ("transfusion", "volume_transfused"),
])
def test_ranges_cover_generated_values(v3, v21, version, table, column):
    cov = range_coverage(v3 if version == "3.0" else v21, version)
    assert cov.get((table, column), 0) == 1.0, f"{table}.{column}: {cov.get((table, column))}"


@pytest.mark.parametrize("version", ["3.0", "2.1"])
def test_med_ranges_apply(v3, v21, version):
    cov = range_coverage(v3 if version == "3.0" else v21, version)
    assert cov[("medication_admin_continuous", "med_dose")] > 0.3


# --------------------------------------------------------------------------
# Each check catches an injected error
# --------------------------------------------------------------------------
class TestMutations:
    def test_schema_missing_column(self, v3):
        t = _copy(v3)
        t["vitals"] = t["vitals"].drop(columns=["vital_category"])
        _assert_caught(t, "schema", table="vitals")

    def test_schema_bad_category(self, v3):
        t = _copy(v3)
        t["vitals"].loc[t["vitals"].index[0], "vital_category"] = "blood_pressure"
        _assert_caught(t, "schema", table="vitals")

    def test_schema_wrong_dtype(self, v3):
        t = _copy(v3)
        t["vitals"]["vital_value"] = t["vitals"]["vital_value"].astype(str)
        _assert_caught(t, "schema", table="vitals")

    def test_primary_key_duplicate(self, v3):
        t = _copy(v3)
        t["vitals"] = pd.concat([t["vitals"], t["vitals"].iloc[[0]]], ignore_index=True)
        _assert_caught(t, "primary_key", table="vitals")

    @pytest.mark.parametrize("version", ["3.0", "2.1"])
    def test_primary_key_duplicate_hospitalization(self, v3, v21, version):
        t = _copy(v3 if version == "3.0" else v21)
        h = t["hospitalization"]
        t["hospitalization"] = pd.concat([h, h.iloc[[0]]], ignore_index=True)
        _assert_caught(t, "primary_key", version, table="hospitalization")

    def test_foreign_key_orphan(self, v3):
        t = _copy(v3)
        t["labs"].loc[t["labs"].index[0], "hospitalization_id"] = "no-such-stay"
        _assert_caught(t, "foreign_key", table="labs")

    def test_foreign_key_patient_mismatch(self, v3):
        t = _copy(v3)
        h = t["hospitalization"]
        other = h["patient_id"].loc[h["patient_id"] != h["patient_id"].iloc[0]].iloc[0]
        t["hospitalization"].loc[h.index[0], "patient_id"] = other
        # the hospitalization's patient no longer owns its ADT rows' patient,
        # or (where tables carry no patient_id) its own stays now overlap
        found = _issues(t, "foreign_key") + _issues(t, "overlap")
        assert found

    def test_chronology_adt_out_before_in(self, v3):
        t = _copy(v3)
        a = t["adt"]
        i = a.index[0]
        a.loc[i, "out_dttm"] = a.loc[i, "in_dttm"] - pd.Timedelta(hours=1)
        _assert_caught(t, "chronology", table="adt")

    def test_chronology_lab_result_before_collect(self, v3):
        t = _copy(v3)
        lb = t["labs"]
        i = lb.index[0]
        lb.loc[i, "lab_result_dttm"] = lb.loc[i, "lab_collect_dttm"] - pd.Timedelta(minutes=30)
        _assert_caught(t, "chronology", table="labs")

    def test_overlap_adt(self, v3):
        t = _copy(v3)
        a = t["adt"].sort_values(["hospitalization_id", "in_dttm"])
        multi = a["hospitalization_id"].duplicated(keep="first")
        i = a.index[multi.to_numpy()][0]
        prev = a.index[a.index.get_loc(i) - 1]
        t["adt"].loc[i, "in_dttm"] = a.loc[prev, "in_dttm"] + pd.Timedelta(minutes=1)
        _assert_caught(t, "overlap", table="adt")

    @pytest.mark.parametrize("version", ["3.0", "2.1"])
    def test_stay_window_vitals_after_discharge(self, v3, v21, version):
        t = _copy(v3 if version == "3.0" else v21)
        v = t["vitals"]
        i = v.index[0]
        dis = t["hospitalization"].set_index("hospitalization_id")["discharge_dttm"]
        v.loc[i, "recorded_dttm"] = dis[v.loc[i, "hospitalization_id"]] + pd.Timedelta(days=1)
        _assert_caught(t, "stay_window", version, table="vitals")

    def test_stay_window_before_admission(self, v3):
        t = _copy(v3)
        lb = t["labs"]
        i = lb.index[0]
        adm = t["hospitalization"].set_index("hospitalization_id")["admission_dttm"]
        start = adm[lb.loc[i, "hospitalization_id"]] - pd.Timedelta(days=2)
        lb.loc[i, ["lab_order_dttm", "lab_collect_dttm"]] = start
        _assert_caught(t, "stay_window", table="labs")

    def test_value_range_vital(self, v3):
        t = _copy(v3)
        v = t["vitals"]
        i = v.index[v["vital_category"] == "spo2"][0]
        v.loc[i, "vital_value"] = 150.0
        _assert_caught(t, "value_range", table="vitals")

    def test_value_range_lab(self, v21):
        t = _copy(v21)
        lb = t["labs"]
        i = lb.index[lb["lab_category"].str.lower() == "sodium"][0]
        lb.loc[i, "lab_value_numeric"] = 900.0
        _assert_caught(t, "value_range", "2.1", table="labs")

    def test_value_range_hemodynamics(self, v3):
        t = _copy(v3)
        h = t["invasive_hemodynamics"]
        i = h.index[h["measurement_category"] == "cvp"][0]
        h.loc[i, "measurement_value"] = 500.0
        _assert_caught(t, "value_range", table="invasive_hemodynamics")

    def test_value_range_negative_volume(self, v3):
        t = _copy(v3)
        t["transfusion"].loc[t["transfusion"].index[0], "volume_transfused"] = -10.0
        _assert_caught(t, "value_range", table="transfusion")

    @pytest.mark.parametrize("version", ["3.0", "2.1"])
    def test_units_lab(self, v3, v21, version):
        t = _copy(v3 if version == "3.0" else v21)
        lb = t["labs"]
        i = lb.index[lb["lab_category"].str.lower() == "sodium"][0]
        lb.loc[i, "reference_unit"] = "mg/dL"
        _assert_caught(t, "units", version, table="labs")

    def test_units_continuous_med_not_a_rate(self, v3):
        t = _copy(v3)
        m = t["medication_admin_continuous"]
        m.loc[m.index[0], "med_dose_unit"] = "mg"
        _assert_caught(t, "units", table="medication_admin_continuous")

    def test_death_missing(self, v3):
        t = _copy(v3)
        h = t["hospitalization"]
        dead = h.loc[h["discharge_category"].str.lower() == "expired", "patient_id"]
        assert len(dead), "fixture has no expired discharge"
        p = t["patient"]
        p.loc[p["patient_id"] == dead.iloc[0], "death_dttm"] = pd.NaT
        _assert_caught(t, "death")

    def test_death_mismatch(self, v3):
        t = _copy(v3)
        h = t["hospitalization"]
        dead = h.loc[h["discharge_category"].str.lower() == "expired", "patient_id"].iloc[0]
        p = t["patient"]
        p.loc[p["patient_id"] == dead, "death_dttm"] += pd.Timedelta(days=10)
        found = _issues(t, "death") + _issues(t, "stay_window")
        assert found

    def test_never_charted_hfnc_flow(self, v3):
        t = _copy(v3)
        r = t["respiratory_support"]
        hfnc = r["device_category"] == "hfnc"
        assert hfnc.sum() >= 20, "fixture has too few high flow rows"
        r.loc[hfnc, "lpm_set"] = np.nan
        _assert_caught(t, "never_charted", table="respiratory_support")


def test_new_table_gets_generic_checks(v3):
    """A table the checks have never heard of is still checked on its keys
    and timestamps, so newly generated tables are covered by default."""
    t = _copy(v3)
    h = t["hospitalization"].iloc[:3]
    t["made_up_table"] = pd.DataFrame({
        "hospitalization_id": list(h["hospitalization_id"]) + ["no-such-stay"],
        "event_start_dttm": list(h["admission_dttm"]) + [h["admission_dttm"].iloc[0]],
        "event_end_dttm": list(h["admission_dttm"] - pd.Timedelta(hours=1)) + [h["admission_dttm"].iloc[0]],
    })
    found = {i["check"] for i in check_dataset(t, "3.0", checks=["foreign_key", "chronology"])
             if i["table"] == "made_up_table"}
    assert found == {"foreign_key", "chronology"}
