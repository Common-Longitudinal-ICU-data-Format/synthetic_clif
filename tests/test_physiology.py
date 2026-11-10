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


@pytest.fixture(scope="module")
def big_tables():
    """Larger 2.1 dataset, so rarer events (transfusion, CRRT, PA catheters) occur."""
    ds = SyntheticCLIFDataset(n_patients=220, n_hospitalizations=300, seed=21, clif_version="2.1")
    return ds.generate(verbose=False)


class TestOutcomes:
    def test_death_follows_illness(self, big_tables):
        h = big_tables["hospitalization"]
        died = h["discharge_category"].str.lower() == "expired"
        assert 0.04 < died.mean() < 0.25
        rs = big_tables["respiratory_support"]
        imv = h["hospitalization_id"].isin(set(rs.loc[rs["device_category"] == "IMV", "hospitalization_id"]))
        assert died[imv].mean() > 2 * died[~imv].mean()

    def test_only_last_stay_expires_and_death_time_matches(self, big_tables):
        h = big_tables["hospitalization"].sort_values("admission_dttm")
        p = big_tables["patient"].set_index("patient_id")
        dead = h[h["discharge_category"].str.lower() == "expired"]
        last = h.groupby("patient_id")["hospitalization_id"].last()
        assert dead["hospitalization_id"].isin(set(last)).all()
        dd = pd.to_datetime(p.loc[dead["patient_id"], "death_dttm"], utc=True).to_numpy()
        assert (dd == pd.to_datetime(dead["discharge_dttm"], utc=True).to_numpy()).all()


class TestCareProcesses:
    def test_icu_covers_ventilation(self, big_tables):
        adt = big_tables["adt"]
        icu = adt[adt["location_category"].str.lower() == "icu"]
        rs = big_tables["respiratory_support"]
        imv = rs[rs["device_category"] == "IMV"][["hospitalization_id", "recorded_dttm"]]
        m = imv.merge(icu[["hospitalization_id", "in_dttm", "out_dttm"]], on="hospitalization_id")
        m["inside"] = (m["recorded_dttm"] >= m["in_dttm"]) & (m["recorded_dttm"] <= m["out_dttm"])
        covered = m.groupby(["hospitalization_id", "recorded_dttm"])["inside"].any()
        assert covered.mean() > 0.97

    def test_rass_tracks_sedation_and_cam(self, big_tables):
        pa = big_tables["patient_assessments"]
        rass = pa[pa["assessment_category"] == "RASS"]
        assert len(rass)
        rs = big_tables["respiratory_support"][["hospitalization_id", "recorded_dttm", "device_category"]]
        r = pd.merge_asof(rass.sort_values("recorded_dttm"), rs.sort_values("recorded_dttm"),
                          on="recorded_dttm", by="hospitalization_id", direction="backward",
                          tolerance=pd.Timedelta(hours=2))
        on = r["device_category"] == "IMV"
        assert r.loc[on, "numerical_value"].mean() < r.loc[~on, "numerical_value"].mean() - 1
        cam = pa[pa["assessment_category"].str.contains("cam", case=False)]
        c = pd.merge_asof(cam.sort_values("recorded_dttm"),
                          rass[["hospitalization_id", "recorded_dttm", "numerical_value"]]
                          .rename(columns={"numerical_value": "rass"}).sort_values("recorded_dttm"),
                          on="recorded_dttm", by="hospitalization_id", direction="nearest",
                          tolerance=pd.Timedelta(minutes=5)).dropna(subset=["rass"])
        deep = c["rass"] <= -4
        assert deep.any()
        assert c.loc[deep, "categorical_value"].astype(str).str.contains("unable", case=False).mean() > 0.8

    def test_transfusion_follows_low_hemoglobin(self, big_tables):
        tr = big_tables["transfusion"]
        rbc = tr[tr["component_name"] == "Packed RBCs"].sort_values("transfusion_start_dttm")
        assert len(rbc) > 5
        lb = big_tables["labs"]
        hg = lb[lb["lab_category"] == "hemoglobin"].sort_values("lab_collect_dttm")
        m = pd.merge_asof(rbc, hg[["hospitalization_id", "lab_collect_dttm", "lab_value_numeric"]],
                          left_on="transfusion_start_dttm", right_on="lab_collect_dttm",
                          by="hospitalization_id", direction="backward").dropna(subset=["lab_value_numeric"])
        assert m["lab_value_numeric"].median() < 8.0
        assert hg["lab_value_numeric"].median() > 9.0

    def test_blood_products_match_intake(self, big_tables):
        tr = big_tables["transfusion"]
        io = big_tables["intake_output"]
        blood = io[io["fluid_name"].isin(set(tr["component_name"]))]
        assert blood["amount"].sum() == pytest.approx(tr["volume_transfused"].sum(), rel=0.01)

    def test_antibiotics_follow_cultures(self, big_tables):
        mi = big_tables["medication_admin_intermittent"]
        abx = mi[mi["med_group"] == "CMS_sepsis_qualifying_antibiotics"]
        cult = big_tables["microbiology_culture"]
        first_abx = abx.groupby("hospitalization_id")["admin_dttm"].min()
        first_cx = cult.groupby("hospitalization_id")["collect_dttm"].min()
        both = pd.concat([first_abx, first_cx], axis=1, keys=["abx", "cx"]).dropna()
        assert len(both) > 10
        # cultures are drawn before (or within an hour of) the first dose for most stays
        assert ((both["cx"] - both["abx"]) < pd.Timedelta(hours=1)).mean() > 0.8

    def test_hemodynamics_only_with_pa_catheter(self, big_tables):
        ih = big_tables["invasive_hemodynamics"]
        if ih.empty:
            pytest.skip("no PA catheters in this sample")
        w = ih.pivot_table(index=["hospitalization_id", "recorded_dttm"], columns="measure_name",
                           values="measure_value")
        assert (w["PA Mean"] > w["PCWP"]).all()
        ci = w["Cardiac Index"]
        svr = w["SVR"]
        assert np.corrcoef(ci, svr)[0, 1] < -0.3

    def test_med_orders_match_administrations(self, big_tables):
        mo = big_tables["medication_orders"].set_index("med_order_id")
        mi = big_tables["medication_admin_intermittent"]
        mc = big_tables["medication_admin_continuous"]
        ids = set(mi["med_order_id"]) | set(mc["med_order_id"])
        assert ids == set(mo.index)
        span = pd.concat([mi, mc]).groupby("med_order_id")["admin_dttm"].agg(["min", "max"])
        j = span.join(mo[["order_start_dttm", "order_end_dttm"]])
        assert (j["order_start_dttm"] == j["min"]).all()
        assert (j["order_end_dttm"] >= j["max"] - pd.Timedelta(minutes=1)).all()
        adm = pd.concat([mi, mc]).merge(big_tables["hospitalization"][["hospitalization_id", "discharge_dttm"]],
                                        on="hospitalization_id")
        assert (adm["admin_dttm"] <= adm["discharge_dttm"]).all()


class TestPatientLevel:
    def test_height_constant_across_stays(self, big_tables):
        v = big_tables["vitals"]
        ht = v[v["vital_category"] == "height_cm"].merge(
            big_tables["hospitalization"][["hospitalization_id", "patient_id"]], on="hospitalization_id")
        per_stay = ht.groupby(["patient_id", "hospitalization_id"])["vital_value"].median()
        spread = per_stay.groupby(level=0).agg(lambda x: x.max() - x.min())
        multi = per_stay.groupby(level=0).size() > 1
        if not multi.any():
            pytest.skip("no readmissions")
        # charted heights include the occasional entry error, so judge the typical patient
        assert (spread[multi] < 3).mean() > 0.9

    def test_problem_list_matches_course(self, big_tables):
        pdx = big_tables["patient_diagnosis"]
        ckd = set(pdx.loc[pdx["diagnosis_code"] == "N18.9", "patient_id"])
        lb = big_tables["labs"].merge(big_tables["hospitalization"][["hospitalization_id", "patient_id"]],
                                      on="hospitalization_id")
        cr = lb[lb["lab_category"] == "creatinine"].groupby("patient_id")["lab_value_numeric"].min()
        assert cr[cr.index.isin(ckd)].median() > 1.3 * cr[~cr.index.isin(ckd)].median()
        assert pdx["start_dttm"].max() < pd.Timestamp("2025-01-01", tz="UTC")


class TestOrders:
    def test_consults_follow_course(self, big_tables):
        k = big_tables["key_icu_orders"]
        h = big_tables["hospitalization"]
        dead = set(h.loc[h["discharge_category"].str.lower() == "expired", "hospitalization_id"])
        pal = set(k.loc[k["order_category"] == "Palliative Care Consult", "hospitalization_id"])
        assert len(pal & dead) / max(len(dead), 1) > 0.4
        assert len(pal - dead) / (len(h) - len(dead)) < 0.1

    def test_therapy_sessions_follow_orders(self, big_tables):
        th = big_tables["therapy_details"]
        k = big_tables["key_icu_orders"]
        pt = th[th["therapy_element_category"] == "Physical Therapy"].groupby("hospitalization_id")["session_start_dttm"].min()
        po = k[k["order_category"] == "PT Evaluation"].groupby("hospitalization_id")["order_dttm"].min()
        assert set(pt.index) <= set(po.index)
        assert (pt >= po.reindex(pt.index)).all()
