"""CLIF 3.0 ``drain`` table (tubes and drains).

Every drain follows something that already happened in the stay:

- **foley**: placed on arrival in the ICU (often in the ED just before) and
  removed near ICU discharge; a few ward patients get one for retention.
  Urine output charted while it is in place is labelled
  ``indwelling_urinary_catheter`` in ``output`` and carries its ``lda_id``.
- **eud**: an external urinary device for some patients after the Foley
  comes out on the ward; urine charted then is ``external_urinary_catheter``.
- **og_tube / ng_tube**: placed at intubation for each invasive-ventilation
  episode and removed at extubation.
- **dobhoff**: a post-pyloric feeding tube for some patients when tube
  feeds start.
- **peg_tube**: a gastrostomy a few days after tracheostomy (course event,
  also in ``patient_procedures``).
- **chest_tube / pigtail**: one per chest-tube procedure in
  ``patient_procedures``; its drainage is charted in ``output``.
- **jp_drain**: source control for an abdominal infection (course event,
  also in ``patient_procedures``).

External ventricular drains are not generated: the course has no
neurosurgical patients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import StayLocations, at, iter_courses, stable_id, stream

CHEST_TUBE_CODES = {"0W9930Z", "0W9940Z", "32551"}


class DrainGenerator(BaseGenerator):
    COLUMNS = ["hospitalization_id", "drain_id", "placed_dttm", "removed_dttm",
               "drain_category", "anatomical_location"]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        adt_df: pd.DataFrame | None = None,
        procedures_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (drain rows, drain output rows for the ``output`` table)."""
        locs = StayLocations(adt_df)
        chest = {}
        if procedures_df is not None and len(procedures_df):
            p = procedures_df[procedures_df["procedure_code"].astype(str).isin(CHEST_TUBE_CODES)]
            for hid, g in p.groupby("hospitalization_id"):
                chest[hid] = sorted(pd.to_datetime(g["procedure_billed_dttm"], utc=True))

        rows, out = [], []
        for row, c, admit, discharge in iter_courses(hospitalizations_df):
            hid = row["hospitalization_id"]
            r = stream("drain", c.rng_seed)
            expired = c.expired
            last = discharge - pd.Timedelta(1, unit="h")

            def add(cat, placed, removed, site):
                placed = max(placed, admit)
                if placed >= discharge:
                    return None
                if removed is not None and (pd.isna(removed) or removed >= discharge):
                    # still in place: at death, or a device the patient goes home with
                    removed = pd.NaT if (expired or cat == "peg_tube") else last
                if removed is not None and pd.notna(removed) and removed <= placed:
                    removed = placed + pd.Timedelta(float(r.uniform(1, 4)), unit="h")
                    if removed >= discharge:
                        removed = pd.NaT if expired else last
                k = sum(1 for x in rows if x["hospitalization_id"] == hid and x["drain_category"] == cat)
                did = f"DRN-{stable_id(hid, cat, k, n=8).upper()}"
                rows.append({"hospitalization_id": hid, "drain_id": did, "placed_dttm": placed,
                             "removed_dttm": removed, "drain_category": cat,
                             "anatomical_location": site})
                return did

            hours = lambda lo, hi: pd.Timedelta(float(r.uniform(lo, hi)), unit="h")

            # Urinary catheters
            icu = locs.icu_windows(hid)
            foley_end = []
            for a, b in icu:
                placed = a - hours(0, 3)
                if pd.isna(b) or b >= discharge - pd.Timedelta(30, unit="min"):
                    removed = discharge
                elif r.random() < 0.7:
                    removed = max(b - hours(0, 12), placed + pd.Timedelta(6, unit="h"))
                else:
                    removed = b + hours(0, 12)
                if add("foley", placed, removed, "Urethra"):
                    foley_end.append(rows[-1]["removed_dttm"])
            if not icu and r.random() < 0.05:
                placed = admit + hours(0, 24)
                if add("foley", placed, placed + hours(24, 96), "Urethra"):
                    foley_end.append(rows[-1]["removed_dttm"])
            for e in foley_end:
                if pd.notna(e) and discharge - e > pd.Timedelta(24, unit="h") \
                        and r.random() < 0.15:
                    placed = e + hours(1, 12)
                    add("eud", placed, placed + hours(24, 96), "External")

            # Gastric and feeding tubes with each invasive-ventilation episode
            for s, e in c.imv_episodes:
                nasal = r.random() < 0.15
                end = at(admit, e) if e < c.los_h - 0.5 else discharge
                add("ng_tube" if nasal else "og_tube", at(admit, s + r.uniform(0.2, 2.0)), end,
                    str(r.choice(["Right naris", "Left naris"])) if nasal else "Mouth")
                fed = np.nonzero(c.tube_feed[s:e] > 0)[0]
                if len(fed) and r.random() < 0.15:
                    add("dobhoff", at(admit, s + fed[0] - r.uniform(1, 4)), end,
                        str(r.choice(["Right naris", "Left naris"])))
            if c.peg_h is not None:
                add("peg_tube", at(admit, c.peg_h), discharge, "Abdomen, left upper quadrant")
            if c.abd_drain is not None:
                s, e = c.abd_drain
                add("jp_drain", at(admit, s), at(admit, e) if e is not None else discharge,
                    str(r.choice(["Right lower quadrant", "Left lower quadrant", "Pelvis",
                                  "Right upper quadrant"])))

            # Chest tubes from the procedures table, with their drainage
            for ts in chest.get(hid, []):
                cat = "pigtail" if r.random() < 0.4 else "chest_tube"
                side = str(r.choice(["Right", "Left"]))
                placed = ts - hours(0, 0.5)
                removed = placed + hours(48, 168)
                did = add(cat, placed, removed, f"{side} pleural space")
                if did is None:
                    continue
                stop = rows[-1]["removed_dttm"]
                stop = discharge if pd.isna(stop) else stop
                vol = r.uniform(200, 600)
                t = placed + hours(6, 8)
                while t < stop:
                    out.append({"hospitalization_id": hid, "recorded_dttm": t,
                                "output_name": "Chest Tube Output", "output_category": "chest_tube_temporary",
                                "output_value": float(round(vol, -1)), "output_unit": "mL", "lda_id": did})
                    vol = max(vol * r.uniform(0.6, 0.85), 10)
                    t += pd.Timedelta(8, unit="h")

        df = pd.DataFrame(rows, columns=self.COLUMNS)
        for col in ("placed_dttm", "removed_dttm"):
            df[col] = pd.to_datetime(df[col], utc=True)
        return df, pd.DataFrame(out)


def label_urine_output(output: pd.DataFrame, drains: pd.DataFrame) -> pd.DataFrame:
    """Set urine output category and ``lda_id`` from the urinary devices in place.

    Charted while a Foley is in: ``indwelling_urinary_catheter``; with an
    external device: ``external_urinary_catheter``; otherwise voided
    (``urethral``).
    """
    if output is None or output.empty or drains is None or drains.empty:
        return output
    out = output.copy()
    if "lda_id" not in out.columns:
        out["lda_id"] = pd.NA
    out["lda_id"] = out["lda_id"].astype("object")
    urine = out["output_name"].astype(str).str.startswith("Urine Output")
    out.loc[urine, ["output_category", "output_name"]] = ["urethral", "Urine Output"]
    dev = drains[drains["drain_category"].isin(["foley", "eud"])]
    if dev.empty:
        return out
    o = out.loc[urine, ["hospitalization_id", "recorded_dttm"]].reset_index()
    o["recorded_dttm"] = pd.to_datetime(o["recorded_dttm"], utc=True)
    m = o.merge(dev[["hospitalization_id", "drain_id", "drain_category", "placed_dttm", "removed_dttm"]],
                on="hospitalization_id")
    end = m["removed_dttm"].fillna(pd.Timestamp.max.tz_localize("UTC"))
    m = m[(m["recorded_dttm"] >= m["placed_dttm"]) & (m["recorded_dttm"] <= end)]
    m = m.sort_values("drain_category", ascending=False).drop_duplicates("index")  # foley wins
    cat = m["drain_category"].map({"foley": "indwelling_urinary_catheter", "eud": "external_urinary_catheter"})
    name = m["drain_category"].map({"foley": "Urine Output (Foley)", "eud": "Urine Output (External Catheter)"})
    out.loc[m["index"].to_numpy(), "output_category"] = cat.to_numpy()
    out.loc[m["index"].to_numpy(), "output_name"] = name.to_numpy()
    out.loc[m["index"].to_numpy(), "lda_id"] = m["drain_id"].to_numpy()
    return out
