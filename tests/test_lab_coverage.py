"""Lab coverage and lab-to-lab coherence.

Every CLIF 2.1.0 lab category is generated with its mCIDE unit and order
category, the extra CLIF 3.0.0 categories appear only in 3.0.0 output, and
the newer labs agree with the rest of the course (differential with WBC,
bilirubin fractions, SaO2 with PaO2, ionized with total calcium, fingersticks
with insulin, mixed venous gases with a PA catheter, triglycerides with
propofol).
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_clif.clif3 import schema
from synthetic_clif.config.lab_catalog import CLIF21_LAB_CATEGORIES, CLIF3_ONLY, LABS, V3_CATEGORY
from synthetic_clif.generators.dataset import SyntheticCLIFDataset
from synthetic_clif.physiology.course import severinghaus_spo2

V21_ORDER = {"blood_gas", "bmp", "cbc", "coags", "lft", "misc"}


@pytest.fixture(scope="module")
def v21():
    ds = SyntheticCLIFDataset(n_patients=100, n_hospitalizations=120, seed=7, clif_version="2.1")
    return ds.generate(verbose=False)


@pytest.fixture(scope="module")
def v3():
    ds = SyntheticCLIFDataset(n_patients=100, n_hospitalizations=120, seed=7, clif_version="3.0")
    return ds.generate(verbose=False)


def _wide(labs, cats):
    x = labs[labs["lab_category"].isin(cats)]
    return x.pivot_table(index=["hospitalization_id", "lab_collect_dttm"], columns="lab_category",
                         values="lab_value_numeric", aggfunc="first")


class TestCoverage:
    def test_catalog_matches_mcide_2_1(self):
        assert len(CLIF21_LAB_CATEGORIES) == 52
        assert not set(CLIF21_LAB_CATEGORIES) & CLIF3_ONLY

    def test_all_2_1_categories_generated(self, v21):
        cats = set(v21["labs"]["lab_category"])
        assert cats == set(CLIF21_LAB_CATEGORIES)

    def test_2_1_units_and_order_categories(self, v21):
        lb = v21["labs"]
        for cat, g in lb.groupby("lab_category"):
            assert set(g["reference_unit"]) == {LABS[cat].unit_21}, cat
        assert set(lb["lab_order_category"]) <= V21_ORDER
        assert set(lb["lab_specimen_category"]) == {"blood/plasma/serum"}

    def test_3_0_categories_units_and_order(self, v3):
        lb = v3["labs"]
        pv = set(schema.permissible_values("labs", "lab_category"))
        cats = set(lb["lab_category"])
        assert cats <= pv
        expected = {V3_CATEGORY.get(c, c) for c in LABS}
        assert cats == expected
        units = {V3_CATEGORY.get(c, c): s.unit_30 for c, s in LABS.items()}
        for cat, g in lb.groupby("lab_category"):
            assert set(g["reference_unit"]) == {units[cat]}, cat
        assert set(lb["lab_order_category"]) <= set(schema.permissible_values("labs", "lab_order_category"))
        glucose = lb[lb["lab_category"] == "glucose"]
        assert set(glucose["lab_order_category"]) == {"bmp", "poc"}

    def test_3_0_only_labs_left_out_of_2_1(self, v21, v3):
        assert not set(v21["labs"]["lab_category"]) & CLIF3_ONLY
        assert CLIF3_ONLY <= set(v3["labs"]["lab_category"])

    def test_blood_gas_bicarbonate(self, v3):
        lb = v3["labs"]
        hco3 = lb[lb["lab_category"] == "bicarbonate"]
        gas = hco3["lab_order_name"].str.contains("Blood Gas")
        assert (hco3.loc[gas, "lab_order_category"] == "blood_gas").all()
        assert (hco3.loc[~gas, "lab_order_category"] == "bmp").all()


class TestLabLinks:
    def test_differential_sums_and_matches_wbc(self, v21):
        diff = [f"{c}_percent" for c in ("neutrophils", "lymphocytes", "monocytes", "eosinophils", "basophils")]
        w = _wide(v21["labs"], diff + ["wbc", "neutrophils_absolute", "lymphocytes_absolute"]).dropna()
        assert len(w) > 100
        assert (w[diff].sum(axis=1) - 100).abs().max() <= 1.5
        for cell in ("neutrophils", "lymphocytes"):
            calc = w["wbc"] * w[f"{cell}_percent"] / 100
            assert (w[f"{cell}_absolute"] - calc).abs().max() <= 0.1

    def test_infection_shifts_differential(self, v21):
        lb = v21["labs"]
        abx = v21["medication_admin_intermittent"]
        treated = set(abx.loc[abx["med_group"].astype(str).str.lower().eq("cms_sepsis_qualifying_antibiotics")
                              | abx["med_category"].isin(["vancomycin", "piperacillin_tazobactam", "cefepime",
                                                          "meropenem", "ceftriaxone"]),
                              "hospitalization_id"])
        neut = lb[lb["lab_category"] == "neutrophils_percent"]
        on = neut["hospitalization_id"].isin(treated)
        assert on.any() and (~on).any()
        assert neut.loc[on, "lab_value_numeric"].mean() > neut.loc[~on, "lab_value_numeric"].mean() + 3

    def test_bilirubin_fractions_add_up(self, v21):
        w = _wide(v21["labs"], ["bilirubin_total", "bilirubin_conjugated", "bilirubin_unconjugated"]).dropna()
        assert len(w) > 20
        parts = w["bilirubin_conjugated"] + w["bilirubin_unconjugated"]
        assert (parts - w["bilirubin_total"]).abs().max() <= 0.15

    def test_sao2_follows_pao2(self, v21):
        w = _wide(v21["labs"], ["po2_arterial", "so2_arterial"]).dropna()
        assert len(w) > 100
        expected = np.minimum(severinghaus_spo2(w["po2_arterial"].to_numpy()), 100)
        assert np.median(np.abs(w["so2_arterial"] - expected)) < 1.5

    def test_ionized_calcium_tracks_corrected_calcium(self, v21):
        lb = v21["labs"]
        ica = lb[lb["lab_category"] == "calcium_ionized"].dropna(subset=["lab_value_numeric"])
        assert 4.3 < ica["lab_value_numeric"].median() < 5.2
        cmp = _wide(lb, ["calcium_total", "albumin"]).dropna().reset_index()
        cmp["corrected"] = cmp["calcium_total"] + 0.8 * (4.0 - cmp["albumin"])
        m = pd.merge_asof(ica.sort_values("lab_collect_dttm"), cmp.sort_values("lab_collect_dttm"),
                          on="lab_collect_dttm", by="hospitalization_id", direction="nearest",
                          tolerance=pd.Timedelta(hours=8)).dropna(subset=["corrected"])
        assert len(m) > 30
        assert np.corrcoef(m["lab_value_numeric"], m["corrected"])[0, 1] > 0.3

    def test_fingersticks_follow_insulin_and_serum_glucose(self, v21):
        lb = v21["labs"]
        fs = lb[lb["lab_category"] == "glucose_fingerstick"].dropna(subset=["lab_value_numeric"])
        cont = v21["medication_admin_continuous"]
        ins = cont[cont["med_category"] == "insulin"]
        span = ins.groupby("hospitalization_id")["admin_dttm"].agg(["min", "max"])
        j = fs.join(span, on="hospitalization_id")
        during = (j["lab_collect_dttm"] >= j["min"]) & (j["lab_collect_dttm"] <= j["max"])
        hours_on = (span["max"] - span["min"]).dt.total_seconds().sum() / 3600
        assert hours_on > 10
        assert during.sum() / hours_on > 0.5          # about hourly on an insulin drip
        serum = lb[lb["lab_category"] == "glucose_serum"].dropna(subset=["lab_value_numeric"])
        m = pd.merge_asof(fs.sort_values("lab_collect_dttm"),
                          serum[["hospitalization_id", "lab_collect_dttm", "lab_value_numeric"]]
                          .sort_values("lab_collect_dttm").rename(columns={"lab_value_numeric": "serum"}),
                          on="lab_collect_dttm", by="hospitalization_id", direction="nearest",
                          tolerance=pd.Timedelta(hours=1)).dropna(subset=["serum"])
        assert len(m) > 20
        assert np.corrcoef(m["lab_value_numeric"], m["serum"])[0, 1] > 0.6

    def test_mixed_venous_gas_needs_pa_catheter(self, v21):
        lb = v21["labs"]
        svo2 = lb[lb["lab_category"] == "so2_mixed_venous"]
        assert len(svo2) > 0
        assert set(svo2["hospitalization_id"]) <= set(v21["invasive_hemodynamics"]["hospitalization_id"])

    def test_one_troponin_assay_per_stay(self, v21):
        lb = v21["labs"]
        trop = lb[lb["lab_category"].isin(["troponin_i", "troponin_t"])]
        assert trop.groupby("hospitalization_id")["lab_category"].nunique().max() == 1

    def test_hematocrit_is_three_times_hemoglobin(self, v3):
        w = _wide(v3["labs"], ["hemoglobin", "hematocrit"]).dropna()
        assert len(w) > 100
        assert (w["hematocrit"] - 3 * w["hemoglobin"]).abs().median() < 1.0

    def test_triglycerides_only_on_propofol(self, v3):
        lb = v3["labs"]
        tg = set(lb.loc[lb["lab_category"] == "triglycerides", "hospitalization_id"])
        cont = v3["medication_admin_continuous"]
        prop = set(cont.loc[cont["med_category"] == "propofol", "hospitalization_id"])
        assert tg and tg <= prop
