"""CLIF 3.0 ``clinical_notes_facts`` table (note metadata, no text).

Notes are written when the stay calls for them:

- an H&P at admission and a progress note each hospital day, by the ICU
  team while the patient is in the ICU and by hospital medicine otherwise
  (residents' and fellows' notes are co-signed by the attending);
- consult and therapy notes after each consult order (PT/OT/speech,
  nutrition, palliative care, social work), nephrology when dialysis starts,
  infectious diseases after a positive blood culture;
- a procedure note for each bedside procedure in ``patient_procedures``,
  written by its performing provider;
- transfer notes on moves into and out of the ICU (from ``adt``);
- nursing notes each shift and pharmacy notes while vancomycin is dosed;
- a discharge summary and patient instructions for survivors, a death
  pronouncement for patients who die.

Each note has one row per revision: an optional incomplete draft, the
signature, a co-signature when there is a co-signer, and occasional
addenda. Authors come from the ``provider`` table where it lists someone in
that role for the stay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import StayLocations, at, iter_courses, stable_id, stream

CONSULTS = {
    # order_category -> (note_type, note_type_category, specialty, provider role)
    "PT Evaluation": ("PT Evaluation", "therapy_note", "Physical Therapy", "Physical Therapist"),
    "OT Evaluation": ("OT Evaluation", "therapy_note", "Occupational Therapy", "Occupational Therapist"),
    "Speech Evaluation": ("Speech Therapy Evaluation", "therapy_note", "Speech Language Pathology",
                          "Speech Therapist"),
    "Nutrition Consult": ("Nutrition Assessment", "nutrition_note", "Nutrition", "Dietitian"),
    "Palliative Care Consult": ("Palliative Care Consult", "consult_note", "Palliative Care", "Attending"),
    "Social Work Consult": ("Social Work Assessment", "case_management_note", "Social Work", "Social Worker"),
}

PROCEDURE_NOTES = {
    # code -> (note_type, specialty)
    "0BH17EZ": ("Intubation Procedure Note", "Critical Care Medicine"),
    "31500": ("Intubation Procedure Note", "Critical Care Medicine"),
    "02HV33Z": ("Central Line Procedure Note", "Critical Care Medicine"),
    "36556": ("Central Line Procedure Note", "Critical Care Medicine"),
    "05HY33Z": ("Dialysis Catheter Procedure Note", "Critical Care Medicine"),
    "03HY32Z": ("Arterial Line Procedure Note", "Critical Care Medicine"),
    "36620": ("Arterial Line Procedure Note", "Critical Care Medicine"),
    "02HP32Z": ("Pulmonary Artery Catheter Procedure Note", "Critical Care Medicine"),
    "93503": ("Pulmonary Artery Catheter Procedure Note", "Critical Care Medicine"),
    "0B110F4": ("Operative Note - Tracheostomy", "General Surgery"),
    "31600": ("Operative Note - Tracheostomy", "General Surgery"),
    "0BJ08ZZ": ("Bronchoscopy Procedure Note", "Pulmonary Medicine"),
    "31624": ("Bronchoscopy Procedure Note", "Pulmonary Medicine"),
    "0W9930Z": ("Chest Tube Procedure Note", "Critical Care Medicine"),
    "0W9940Z": ("Chest Tube Procedure Note", "Critical Care Medicine"),
    "32551": ("Chest Tube Procedure Note", "Critical Care Medicine"),
    "0W9B30Z": ("Thoracentesis Procedure Note", "Critical Care Medicine"),
    "32555": ("Thoracentesis Procedure Note", "Critical Care Medicine"),
    "49083": ("Paracentesis Procedure Note", "Hospital Medicine"),
    "0W9G30Z": ("IR Drain Placement Note", "Interventional Radiology"),
    "49406": ("IR Drain Placement Note", "Interventional Radiology"),
    "009U3ZX": ("Lumbar Puncture Procedure Note", "Hospital Medicine"),
    "62270": ("Lumbar Puncture Procedure Note", "Hospital Medicine"),
    "0DH63UZ": ("Operative Note - PEG", "Gastroenterology"),
    "43246": ("Operative Note - PEG", "Gastroenterology"),
}

HOME = {"home", "against medical advice (ama)", "group home", "assisted living"}


class ClinicalNotesFactsGenerator(BaseGenerator):
    COLUMNS = ["hospitalization_id", "author_id", "note_id", "note_type", "note_type_category",
               "note_status", "cosigner_id", "cosigner_specialty", "note_author_specialty",
               "revision_id", "creation_dttm", "revision_dttm", "service_date"]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        adt_df: pd.DataFrame | None = None,
        provider_df: pd.DataFrame | None = None,
        orders_df: pd.DataFrame | None = None,
        procedures_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        locs = StayLocations(adt_df)
        staff = {}
        if provider_df is not None and len(provider_df):
            for (hid, role), g in provider_df.groupby(["hospitalization_id", "provider_role_category"]):
                staff.setdefault(hid, {})[str(role)] = list(g["provider_id"].astype(str))
        orders = {}
        if orders_df is not None and len(orders_df):
            o = orders_df[orders_df["order_status_name"].astype(str) != "Discontinued"]
            for hid, g in o.groupby("hospitalization_id"):
                orders[hid] = list(zip(pd.to_datetime(g["order_dttm"], utc=True), g["order_category"]))
        procs = {}
        if procedures_df is not None and len(procedures_df):
            p = procedures_df[procedures_df["procedure_code"].astype(str).isin(PROCEDURE_NOTES)]
            for hid, g in p.groupby("hospitalization_id"):
                procs[hid] = list(zip(pd.to_datetime(g["procedure_billed_dttm"], utc=True),
                                      g["procedure_code"].astype(str), g["performing_provider_id"].astype(str)))

        rows = []
        for row, c, admit, discharge in iter_courses(hospitalizations_df):
            hid = row["hospitalization_id"]
            r = stream("notes", c.rng_seed)
            team = staff.get(hid, {})
            dispo = str(row.get("discharge_category") or "").strip().lower()
            notes = []  # (creation, note_type, category, author, specialty, cosigner, cosigner specialty)

            def person(role):
                ids = team.get(role)
                if ids:
                    return str(r.choice(ids))
                return f"{role[:3].upper()}-{stable_id(hid, role, n=6).upper()}"

            def physician_note(t, note_type, cat):
                icu = locs.at(hid, t) == "icu"
                spec = "Critical Care Medicine" if icu else "Hospital Medicine"
                attending = person("Attending")
                trainee = None
                if icu and team.get("Fellow") and r.random() < 0.6:
                    trainee = person("Fellow")
                elif team.get("Resident") and r.random() < 0.8:
                    trainee = person("Resident")
                elif (team.get("NP") or team.get("PA")) and r.random() < 0.5:
                    trainee = person("NP" if team.get("NP") else "PA")
                if trainee:
                    notes.append((t, note_type, cat, trainee, spec, attending, spec))
                else:
                    notes.append((t, note_type, cat, attending, spec, None, None))

            physician_note(admit + pd.Timedelta(float(r.uniform(1, 6)), unit="h"), "H&P", "h_and_p_note")
            day = admit.normalize() + pd.Timedelta(1, hours=15, unit="D")
            while day < discharge - pd.Timedelta(2, unit="h"):
                physician_note(day + pd.Timedelta(float(r.uniform(0, 5)), unit="h"), "Progress Note", "progress_note")
                day += pd.Timedelta(1, unit="D")

            for t, cat in orders.get(hid, []):
                if cat not in CONSULTS:
                    continue
                note_type, ncat, spec, role = CONSULTS[cat]
                notes.append((t + pd.Timedelta(float(r.uniform(2, 24)), unit="h"), note_type, ncat,
                              person(role), spec, None, None))
            dialysis = [s["start"] for s in c.ihd_sessions]
            if c.crrt.any():
                dialysis.insert(0, float(np.argmax(c.crrt)))
            if dialysis:
                notes.append((at(admit, dialysis[0] - r.uniform(2, 12)), "Nephrology Consult", "consult_note",
                              person("Nephrology"), "Nephrology", None, None))
            bact = [cx["h"] for cx in c.cultures if cx["fluid"] == "blood" and cx["positive"]
                    and not cx["contaminant"]]
            if bact:
                notes.append((at(admit, bact[0] + r.uniform(24, 72)), "Infectious Diseases Consult",
                              "consult_note", person("Infectious Diseases"), "Infectious Diseases", None, None))

            for t, code, performer in procs.get(hid, []):
                note_type, spec = PROCEDURE_NOTES[code]
                notes.append((t + pd.Timedelta(float(r.uniform(0.1, 2)), unit="h"), note_type, "procedure_note",
                              performer, spec, None, None))

            segs = locs.segments(hid)
            for (a0, b0, l0), (a1, b1, l1) in zip(segs, segs[1:]):
                if (l0 == "icu") != (l1 == "icu"):
                    t = a1 + pd.Timedelta(float(r.uniform(-1, 1.5)), unit="h")
                    physician_note(t, "ICU Transfer Note" if l1 == "icu" else "Transfer Note - ICU to Floor",
                                   "transfer_note")

            shift = admit.floor("12h") + pd.Timedelta(3, unit="h")
            while shift < discharge:
                p = 0.9 if locs.at(hid, shift) == "icu" else 0.6
                t = shift + pd.Timedelta(float(r.uniform(6, 11.5)), unit="h")
                if shift >= admit and t < discharge and r.random() < p:
                    notes.append((t, "Nursing Shift Note", "nursing_note", person("RN"), "Nursing", None, None))
                shift += pd.Timedelta(12, unit="h")

            for abx in c.antibiotics:
                if abx["agent"] != "vancomycin":
                    continue
                h = abx["start"] + r.uniform(1, 6)
                while h < abx["end"]:
                    notes.append((at(admit, h), "Pharmacy Note - Vancomycin Dosing", "pharmacy_note",
                                  person("Pharmacist"), "Pharmacy", None, None))
                    h += 24 + r.uniform(-2, 2)

            if c.expired:
                physician_note(discharge + pd.Timedelta(float(r.uniform(5, 60)), unit="min"),
                               "Death Pronouncement", "death_pronouncement")
            else:
                physician_note(discharge - pd.Timedelta(float(r.uniform(0.5, 6)), unit="h"),
                               "Discharge Summary", "discharge_summary")
                notes.append((discharge - pd.Timedelta(float(r.uniform(1, 6)), unit="h"), "Discharge Instructions",
                              "patient_instructions", person("RN"), "Nursing", None, None))
                if dispo not in HOME and not any(n[2] == "case_management_note" for n in notes):
                    notes.append((discharge - pd.Timedelta(float(r.uniform(24, 72)), unit="h"),
                                  "Case Management Discharge Planning", "case_management_note",
                                  person("Case Manager"), "Case Management", None, None))

            for k, (t, note_type, cat, author, spec, cosigner, cospec) in enumerate(sorted(notes, key=lambda x: x[0])):
                t = max(t, admit + pd.Timedelta(5, unit="min"))
                if cat != "death_pronouncement":
                    t = min(t, discharge - pd.Timedelta(5, unit="min"))
                note_id = f"NOTE-{stable_id(hid, k, note_type, n=10).upper()}"
                service = (discharge if cat == "discharge_summary" else t).tz_convert("UTC").normalize().tz_localize(None)
                revs = []
                rt = t
                if r.random() < 0.25:
                    revs.append(("incomplete", rt))
                    rt = rt + pd.Timedelta(float(r.uniform(0.2, 3)), unit="h")
                revs.append(("signed", rt if revs else t + pd.Timedelta(float(r.uniform(5, 90)), unit="min")))
                if cosigner:
                    revs.append(("co-signed", revs[-1][1] + pd.Timedelta(float(r.uniform(1, 18)), unit="h")))
                if r.random() < 0.06:
                    revs.append(("addendum", revs[-1][1] + pd.Timedelta(float(r.uniform(1, 72)), unit="h")))
                for j, (status, rdt) in enumerate(revs, start=1):
                    rows.append({
                        "hospitalization_id": hid, "author_id": author, "note_id": note_id,
                        "note_type": note_type, "note_type_category": cat, "note_status": status,
                        "cosigner_id": cosigner, "cosigner_specialty": cospec,
                        "note_author_specialty": spec, "revision_id": f"{note_id}-{j}",
                        "creation_dttm": t, "revision_dttm": rdt, "service_date": service,
                    })
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        for col in ("creation_dttm", "revision_dttm"):
            df[col] = pd.to_datetime(df[col], utc=True)
        df["service_date"] = pd.to_datetime(df["service_date"])
        return df
