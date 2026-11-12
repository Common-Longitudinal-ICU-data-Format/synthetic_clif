"""Tests for the eight tables new in CLIF 3.0.0.

``intermittent_dialysis``, ``drain``, ``patient_attributes``, ``radiology``,
``clinical_notes_facts``, ``validated_diagnosis``, ``model_registry`` and
``scores`` are generated natively from each stay's shared course
(``synthetic_clif.generators.v3``). These tests check that they conform to the
vendored 3.0 schema and that they agree with the tables they are tied to.
"""

import pandas as pd
import pytest

from synthetic_clif.clif3.validate import validate_table
from synthetic_clif.generators.dataset import SyntheticCLIFDataset
from synthetic_clif.generators.v3.patient_attributes import ALLOWED_VALUES
from synthetic_clif.generators.v3.scores import REGISTRY, _version

NEW_TABLES = [
    "intermittent_dialysis",
    "drain",
    "patient_attributes",
    "radiology",
    "clinical_notes_facts",
    "validated_diagnosis",
    "model_registry",
    "scores",
]


@pytest.fixture(scope="module")
def tables():
    # large enough that rarer events (intermittent haemodialysis, chest tubes)
    # appear at least once with this seed
    ds = SyntheticCLIFDataset(n_patients=225, n_hospitalizations=300, seed=42, clif_version="3.0")
    return ds.generate(verbose=False)


@pytest.fixture(scope="module")
def stays(tables):
    return tables["hospitalization"].set_index("hospitalization_id")[
        ["patient_id", "admission_dttm", "discharge_dttm", "discharge_category"]
    ]


def _with_stay(df, stays):
    return df.join(stays[["admission_dttm", "discharge_dttm"]], on="hospitalization_id")


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", NEW_TABLES)
def test_table_present_and_valid(tables, name):
    assert name in tables
    assert len(tables[name]) > 0
    assert validate_table(name, tables[name]) == []


def test_not_generated_for_2_1():
    ds = SyntheticCLIFDataset(n_patients=5, n_hospitalizations=6, seed=1, clif_version="2.1")
    out = ds.generate(verbose=False)
    assert not set(NEW_TABLES) & set(out)


# --------------------------------------------------------------------------
# intermittent_dialysis
# --------------------------------------------------------------------------
class TestIntermittentDialysis:
    def test_within_stay(self, tables, stays):
        m = _with_stay(tables["intermittent_dialysis"], stays)
        assert (m.recorded_dttm >= m.admission_dttm).all()
        assert (m.recorded_dttm <= m.discharge_dttm).all()

    def test_has_catheter_and_procedure(self, tables):
        ihd = set(tables["intermittent_dialysis"].hospitalization_id)
        line = tables["line"]
        cath = set(line[line.line_category == "dialysis_catheter"].hospitalization_id)
        proc = tables["patient_procedures"]
        billed = set(proc[proc.procedure_code.isin(["5A1D70Z", "90935"])].hospitalization_id)
        assert ihd <= cath
        assert ihd <= billed

    def test_not_on_crrt(self, tables):
        ihd = set(tables["intermittent_dialysis"].hospitalization_id)
        assert not ihd & set(tables["crrt_therapy"].hospitalization_id)

    def test_ultrafiltration_charted_as_output(self, tables):
        ihd = set(tables["intermittent_dialysis"].hospitalization_id)
        out = tables["output"]
        uf = set(out[out.output_category == "ultrafiltration"].hospitalization_id)
        assert ihd <= uf


# --------------------------------------------------------------------------
# drain
# --------------------------------------------------------------------------
class TestDrain:
    def test_within_stay(self, tables, stays):
        m = _with_stay(tables["drain"], stays)
        assert (m.placed_dttm >= m.admission_dttm).all()
        assert (m.placed_dttm <= m.discharge_dttm).all()
        removed = m.dropna(subset=["removed_dttm"])
        assert (removed.removed_dttm >= removed.placed_dttm).all()
        assert (removed.removed_dttm <= removed.discharge_dttm).all()

    def test_gastric_tube_with_every_intubation(self, tables):
        intubated = set(tables["airway"].hospitalization_id)
        d = tables["drain"]
        gastric = set(d[d.drain_category.isin(["og_tube", "ng_tube"])].hospitalization_id)
        assert intubated and intubated <= gastric

    def test_chest_tube_matches_procedure(self, tables):
        d = tables["drain"]
        tubes = set(d[d.drain_category.isin(["chest_tube", "pigtail"])].hospitalization_id)
        proc = tables["patient_procedures"]
        placed = set(proc[proc.procedure_code.isin(["0W9930Z", "0W9940Z", "32551"])].hospitalization_id)
        assert tubes == placed

    def test_urine_output_linked_to_catheter(self, tables):
        out = tables["output"]
        dev = out[out.output_category.isin(["indwelling_urinary_catheter", "external_urinary_catheter"])]
        assert len(dev) > 0
        assert dev.lda_id.notna().all()
        assert set(dev.lda_id) <= set(tables["drain"].drain_id)


# --------------------------------------------------------------------------
# patient_attributes
# --------------------------------------------------------------------------
class TestPatientAttributes:
    def test_values_allowed_for_attribute(self, tables):
        pa = tables["patient_attributes"]
        ok = [v in ALLOWED_VALUES[c] for c, v in zip(pa.attribute_category, pa.attribute_value_category)]
        assert all(ok)

    def test_recorded_before_last_discharge(self, tables, stays):
        pa = tables["patient_attributes"]
        last = stays.groupby("patient_id").discharge_dttm.max()
        assert (pa.recorded_dttm <= pa.patient_id.map(last)).all()

    def test_pregnancy_only_for_women(self, tables):
        pa = tables["patient_attributes"]
        sex = tables["patient"].set_index("patient_id").sex_category
        preg = pa[pa.attribute_category == "pregnancy_status"]
        assert (preg.patient_id.map(sex).str.lower() == "female").all()


# --------------------------------------------------------------------------
# radiology
# --------------------------------------------------------------------------
class TestRadiology:
    def test_times_in_order(self, tables):
        r = tables["radiology"]
        assert (r.radiology_order_dttm <= r.radiology_capture_dttm).all()
        assert (r.radiology_capture_dttm <= r.radiology_report_dttm).all()

    def test_accessions_unique_and_patients_known(self, tables):
        r = tables["radiology"]
        assert not r.radiology_accession.duplicated().any()
        assert r.patient_id.isin(tables["patient"].patient_id).all()

    def test_chest_film_after_intubation(self, tables):
        r = tables["radiology"]
        cxr = r[(r.radiology_modality_category == "xr") & (r.radiology_region_category == "chest")]
        intubated_patients = set(tables["airway"].hospitalization_id.map(
            tables["hospitalization"].set_index("hospitalization_id").patient_id))
        assert intubated_patients <= set(cxr.patient_id)


# --------------------------------------------------------------------------
# clinical_notes_facts
# --------------------------------------------------------------------------
class TestClinicalNotes:
    def test_revisions_in_order(self, tables):
        n = tables["clinical_notes_facts"].sort_values(["note_id", "revision_id"])
        steps = n.groupby("note_id").revision_dttm.diff().dropna()
        assert (steps >= pd.Timedelta(0, unit="s")).all()

    def test_death_note_iff_expired(self, tables, stays):
        n = tables["clinical_notes_facts"]
        expired = set(stays[stays.discharge_category.str.lower() == "expired"].index)
        death = set(n[n.note_type_category == "death_pronouncement"].hospitalization_id)
        summary = set(n[n.note_type_category == "discharge_summary"].hospitalization_id)
        assert death == expired
        assert not summary & expired

    def test_physician_authors_in_provider_table(self, tables):
        n = tables["clinical_notes_facts"]
        phys = n[n.note_type_category.isin(["progress_note", "h_and_p_note"])]
        assert phys.author_id.isin(tables["provider"].provider_id).all()

    def test_within_stay(self, tables, stays):
        m = _with_stay(tables["clinical_notes_facts"], stays)
        assert (m.creation_dttm >= m.admission_dttm - pd.Timedelta(1, unit="h")).all()
        assert (m.creation_dttm <= m.revision_dttm).all()


# --------------------------------------------------------------------------
# validated_diagnosis
# --------------------------------------------------------------------------
class TestValidatedDiagnosis:
    def test_confirmed_sepsis_is_billed(self, tables):
        v = tables["validated_diagnosis"]
        sepsis = set(v[(v.diagnosis_category == "sepsis") & (v.diagnosis_status == "confirmed")].hospitalization_id)
        dx = tables["hospital_diagnosis"]
        billed = set(dx[dx.diagnosis_code.astype(str).str.replace(".", "").str.startswith("A41")].hospitalization_id)
        assert sepsis and sepsis <= billed

    def test_confirmed_arf_was_ventilated(self, tables):
        v = tables["validated_diagnosis"]
        arf = set(v[(v.diagnosis_category == "arf") & (v.diagnosis_status == "confirmed")].hospitalization_id)
        rs = tables["respiratory_support"]
        vent = set(rs[rs.device_category.isin(["imv", "nippv", "cpap"])].hospitalization_id)
        assert arf and arf <= vent

    def test_reviewed_after_discharge(self, tables, stays):
        m = _with_stay(tables["validated_diagnosis"], stays)
        assert (m.review_timestamp > m.discharge_dttm).all()
        assert (m.diagnosis_start_dttm <= m.discharge_dttm).all()

    def test_uncertain_goes_to_panel(self, tables):
        v = tables["validated_diagnosis"]
        assert (v[v.diagnosis_status == "uncertain"].validation_method == "consensus_panel").all()


# --------------------------------------------------------------------------
# model_registry and scores
# --------------------------------------------------------------------------
class TestScores:
    def test_registry_ids_unique(self, tables):
        reg = tables["model_registry"]
        assert not reg.model_id.duplicated().any()
        assert len(reg) == len(REGISTRY)

    def test_scores_use_live_version(self, tables):
        s = tables["scores"]
        reg = tables["model_registry"].set_index("model_id")
        assert s.model_id.isin(reg.index).all()
        released = pd.to_datetime(s.model_id.map(reg.release_date)).dt.tz_localize("UTC")
        assert (s.recorded_dttm >= released).all()
        sample = s.sample(500, random_state=0)
        cats = sample.model_id.map(reg.model_category)
        assert all(_version(c, t) == m for c, t, m in zip(cats, sample.recorded_dttm, sample.model_id))

    def test_scores_in_range_and_within_stay(self, tables, stays):
        m = _with_stay(tables["scores"], stays)
        assert m.score_value.between(0, 100).all()
        assert (m.recorded_dttm <= m.discharge_dttm).all()

    def test_deterioration_index_higher_in_deaths(self, tables, stays):
        s = tables["scores"]
        edi = s[s.model_id.str.startswith("EDI")]
        died = edi.hospitalization_id.map(stays.discharge_category.str.lower() == "expired")
        means = edi.groupby(died).score_value.mean()
        assert means[True] > means[False] + 5
