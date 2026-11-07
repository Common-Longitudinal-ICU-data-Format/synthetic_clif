"""Cross-table physiological coherence (synthetic_clif.physiology).

The time-series generators observe one shared course per hospitalization, so
values that clinicians would expect to agree do agree: MAP with SBP/DBP,
vasopressors with MAP, ventilator settings with each other and with SpO2,
and labs with their own history and with each other.
"""

import numpy as np
import pandas as pd
import pytest

from synthetic_clif.generators.dataset import SyntheticCLIFDataset
from synthetic_clif.physiology import HospitalCourse, course_for_row


@pytest.fixture(scope="module")
def tables():
    ds = SyntheticCLIFDataset(n_patients=60, n_hospitalizations=80, seed=3, clif_version="2.1")
    return ds.generate(verbose=False)


@pytest.fixture(scope="module")
def wide_vitals(tables):
    v = tables["vitals"]
    return v.pivot_table(
        index=["hospitalization_id", "recorded_dttm"],
        columns="vital_category",
        values="vital_value",
        aggfunc="first",
    )


def _lag1(df, value, time, group="hospitalization_id"):
    x = df[[group, time, value]].dropna().sort_values([group, time])
    x["prev"] = x.groupby(group)[value].shift()
    x = x.dropna()
    return np.corrcoef(x[value], x["prev"])[0, 1]


class TestVitals:
    def test_map_derived_from_sbp_dbp(self, wide_vitals):
        bp = wide_vitals.dropna(subset=["sbp", "dbp", "map"])
        assert len(bp) > 100
        assert (bp["sbp"] > bp["dbp"]).all()
        assert ((bp["map"] >= bp["dbp"]) & (bp["map"] <= bp["sbp"])).all()
        calc = bp["dbp"] + (bp["sbp"] - bp["dbp"]) / 3
        assert (bp["map"] - calc).abs().max() <= 8
        assert np.corrcoef(bp["map"], calc)[0, 1] > 0.95

    def test_vitals_are_autocorrelated(self, wide_vitals):
        w = wide_vitals.reset_index()
        assert _lag1(w, "heart_rate", "recorded_dttm") > 0.4
        assert _lag1(w, "map", "recorded_dttm") > 0.4

    def test_spo2_falls_as_fio2_rises(self, tables, wide_vitals):
        sp = wide_vitals["spo2"].dropna().reset_index().sort_values("recorded_dttm")
        rs = tables["respiratory_support"]
        f = rs[["hospitalization_id", "recorded_dttm", "fio2_set"]].dropna().sort_values("recorded_dttm")
        mx = pd.merge_asof(sp, f, on="recorded_dttm", by="hospitalization_id",
                           direction="backward", tolerance=pd.Timedelta(hours=2)).dropna()
        assert np.corrcoef(mx["spo2"], mx["fio2_set"])[0, 1] < -0.1


class TestVasopressors:
    def test_pressors_follow_low_map(self, tables, wide_vitals):
        mac = tables["medication_admin_continuous"]
        pres = mac[mac["med_group"] == "vasoactives"]
        assert len(pres)
        m = wide_vitals["map"].dropna().reset_index()
        first = pres.groupby("hospitalization_id")["admin_dttm"].min().rename("p0")
        mm = m.merge(first, on="hospitalization_id")
        before = mm[(mm["recorded_dttm"] < mm["p0"])
                    & (mm["recorded_dttm"] > mm["p0"] - pd.Timedelta(hours=3))]
        assert before["map"].mean() < 65
        on_pressor = m["hospitalization_id"].isin(set(pres["hospitalization_id"]))
        assert m.loc[on_pressor, "map"].mean() < m.loc[~on_pressor, "map"].mean()

    def test_mar_sequence(self, tables):
        mac = tables["medication_admin_continuous"]
        course_meds = mac[mac["med_category"] != "heparin"]
        for _, g in course_meds.sort_values("admin_dttm").groupby("med_order_id"):
            assert g["mar_action_category"].iloc[0] == "start"
            assert g["mar_action_category"].iloc[-1] == "stop"

    def test_sedation_only_during_imv(self, tables):
        mac = tables["medication_admin_continuous"]
        sed = mac[mac["med_category"].isin(["propofol", "dexmedetomidine", "midazolam"])]
        rs = tables["respiratory_support"]
        imv = rs[rs["device_category"] == "IMV"]
        assert len(sed)
        window = imv.groupby("hospitalization_id")["recorded_dttm"].agg(["min", "max"])
        s = sed.merge(window, left_on="hospitalization_id", right_index=True, how="left")
        assert s["min"].notna().all()
        slack = pd.Timedelta(hours=3)
        assert ((s["admin_dttm"] >= s["min"] - slack) & (s["admin_dttm"] <= s["max"] + slack)).all()


class TestVentilator:
    def test_mechanics(self, tables):
        rs = tables["respiratory_support"]
        imv = rs[rs["device_category"] == "IMV"]
        assert len(imv) > 50
        rr = imv.dropna(subset=["resp_rate_set", "resp_rate_obs"])
        assert (rr["resp_rate_obs"] >= rr["resp_rate_set"]).all()
        pp = imv.dropna(subset=["plateau_pressure_obs", "peak_inspiratory_pressure_obs"])
        assert (pp["plateau_pressure_obs"] <= pp["peak_inspiratory_pressure_obs"]).all()
        assert (pp["plateau_pressure_obs"] >= imv.loc[pp.index, "peep_set"]).all()
        mv = imv.dropna(subset=["minute_vent_obs", "tidal_volume_obs", "resp_rate_obs"])
        calc = mv["tidal_volume_obs"] * mv["resp_rate_obs"] / 1000
        assert (mv["minute_vent_obs"] - calc).abs().max() < 1.5

    def test_settings_persist(self, tables):
        rs = tables["respiratory_support"]
        imv = rs[rs["device_category"] == "IMV"].sort_values(["hospitalization_id", "recorded_dttm"])
        for col in ("tidal_volume_set", "vent_brand_name"):
            x = imv[["hospitalization_id", col]].dropna()
            changed = x[col] != x.groupby("hospitalization_id")[col].shift()
            first = ~x["hospitalization_id"].duplicated()
            assert (changed & ~first).mean() < 0.05, col

    def test_tidal_volume_by_body_size(self, tables):
        rs = tables["respiratory_support"]
        vt = rs.dropna(subset=["tidal_volume_set"])
        assert vt["tidal_volume_set"].between(250, 750).all()


class TestLabs:
    def test_labs_persist(self, tables):
        lb = tables["labs"]
        for cat in ("creatinine", "hemoglobin", "sodium"):
            x = lb[lb["lab_category"] == cat]
            assert _lag1(x, "lab_value_numeric", "lab_collect_dttm") > 0.6, cat

    def test_linked_labs(self, tables):
        lb = tables["labs"]
        w = lb.pivot_table(index=["hospitalization_id", "lab_collect_dttm"],
                           columns="lab_category", values="lab_value_numeric", aggfunc="first")
        p = w.dropna(subset=["bun", "creatinine"])
        assert np.corrcoef(p["bun"], p["creatinine"])[0, 1] > 0.5
        gas = w.dropna(subset=["ph_arterial", "pco2_arterial", "bicarbonate"])
        assert len(gas) > 10
        hh = 6.1 + np.log10(gas["bicarbonate"] / (0.0307 * gas["pco2_arterial"]))
        assert (gas["ph_arterial"] - hh).abs().max() < 0.03

    def test_lactate_higher_on_pressors(self, tables):
        lb = tables["labs"]
        mac = tables["medication_admin_continuous"]
        ph = set(mac.loc[mac["med_group"] == "vasoactives", "hospitalization_id"])
        lac = lb[lb["lab_category"] == "lactate"]
        on = lac["hospitalization_id"].isin(ph)
        assert lac.loc[on, "lab_value_numeric"].mean() > lac.loc[~on, "lab_value_numeric"].mean() + 0.5


class TestCourse:
    def test_deterministic(self):
        row = {"hospitalization_id": "H1", "admission_dttm": pd.Timestamp("2024-01-01", tz="UTC"),
               "discharge_dttm": pd.Timestamp("2024-01-06", tz="UTC"), "_course_seed": 99}
        a = HospitalCourse.build("H1", row["admission_dttm"], row["discharge_dttm"], 99)
        b = course_for_row(row)
        np.testing.assert_array_equal(a.map, b.map)
        np.testing.assert_array_equal(a.fio2, b.fio2)

    def test_same_seed_same_tables(self, tables):
        again = SyntheticCLIFDataset(n_patients=60, n_hospitalizations=80, seed=3,
                                     clif_version="2.1").generate(verbose=False)
        for name in ("vitals", "respiratory_support", "labs"):
            pd.testing.assert_frame_equal(tables[name], again[name], obj=name)

    def test_standalone_generators_agree(self, hospitalizations_df, seed, mcide):
        """Without the orchestrator, generators still share a course per stay."""
        from synthetic_clif.generators.respiratory import RespiratoryGenerator
        from synthetic_clif.generators.medications import MedicationContinuousGenerator

        rs = RespiratoryGenerator(seed=seed, mcide=mcide).generate(hospitalizations_df)
        mac = MedicationContinuousGenerator(seed=seed + 1, mcide=mcide).generate(hospitalizations_df)
        imv = set(rs.loc[rs["device_category"] == "IMV", "hospitalization_id"])
        sed = set(mac.loc[mac["med_group"] == "sedation", "hospitalization_id"]) if len(mac) else set()
        assert sed <= imv
