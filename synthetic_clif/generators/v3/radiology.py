"""CLIF 3.0 ``radiology`` table.

Imaging is ordered for the reasons it is ordered in an ICU:

- a chest film on arrival, after each intubation, after central or dialysis
  line placement, after chest tubes and feeding tubes, and most mornings
  while on invasive ventilation;
- CT of the chest or abdomen and pelvis to find an infection's source, with
  IV contrast unless the kidneys are injured;
- CT head for new coma or delirium; right-upper-quadrant ultrasound after
  shock liver; renal ultrasound for kidney injury; echocardiograms in shock;
  lower-extremity Dopplers in long stays.

Studies on patients in the ICU (or critically ill) are portable when the
modality allows. Order, acquisition and report times follow in that order.
The report text is a short synthetic impression consistent with the course
(tube positions, opacities on severe respiratory failure, effusions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.generators.v3._common import StayLocations, at, critical, iter_courses, stable_id, stream

# (order name, modality, region, contrast, portable-capable)
STUDIES = {
    "cxr": ("XR CHEST 1 VIEW PORTABLE", "xr", "chest", "not_applicable", True),
    "cxr_2v": ("XR CHEST 2 VIEWS", "xr", "chest", "not_applicable", False),
    "kub": ("XR ABDOMEN 1 VIEW", "xr", "abdomen", "not_applicable", True),
    "ct_chest_c": ("CT CHEST W CONTRAST", "ct", "chest", "with_contrast", False),
    "ct_chest": ("CT CHEST WO CONTRAST", "ct", "chest", "without_contrast", False),
    "ct_ap_c": ("CT ABDOMEN PELVIS W CONTRAST", "ct", "abdomen_pelvis", "with_contrast", False),
    "ct_ap": ("CT ABDOMEN PELVIS WO CONTRAST", "ct", "abdomen_pelvis", "without_contrast", False),
    "ct_head": ("CT HEAD WO CONTRAST", "ct", "brain", "without_contrast", False),
    "us_ruq": ("US ABDOMEN LIMITED RUQ", "us", "ruq", "not_applicable", True),
    "us_renal": ("US RENAL COMPLETE", "us", "abdomen", "not_applicable", True),
    "echo": ("ECHO TRANSTHORACIC COMPLETE", "us", "heart", "not_applicable", True),
    "le_doppler": ("US DOPPLER VENOUS LOWER EXTREMITY BILATERAL", "us", "vascular", "not_applicable", True),
    "cta_chest": ("CTA CHEST PE PROTOCOL", "ct", "vascular", "with_contrast", False),
}


class RadiologyGenerator(BaseGenerator):
    COLUMNS = ["patient_id", "radiology_accession", "radiology_order_dttm", "radiology_capture_dttm",
               "radiology_report_dttm", "radiology_order_name", "radiology_modality_category",
               "radiology_region_category", "iv_contrast_category", "radiology_location_category",
               "radiology_report_text"]

    def generate(
        self,
        hospitalizations_df: pd.DataFrame,
        adt_df: pd.DataFrame | None = None,
        drain_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        locs = StayLocations(adt_df)
        tubes = {}
        if drain_df is not None and len(drain_df):
            d = drain_df[drain_df["drain_category"].isin(["chest_tube", "pigtail", "dobhoff", "og_tube", "ng_tube"])]
            for hid, g in d.groupby("hospitalization_id"):
                tubes[hid] = list(zip(g["drain_category"], g["placed_dttm"], g["removed_dttm"]))
        rows = []
        for row, c, admit, discharge in iter_courses(hospitalizations_df):
            hid, pid = row["hospitalization_id"], row.get("patient_id")
            r = stream("radiology", c.rng_seed)
            n = c.n_hours
            crit = critical(c)
            orders = []  # (hour, study key, reason)

            def order(h, key, why):
                if 0 <= h < c.los_h - 0.5:
                    orders.append((float(h), key, why))

            if r.random() < 0.85:
                order(r.uniform(0, 3), "cxr" if crit[: 6].any() or r.random() < 0.6 else "cxr_2v", "admit")
            for s, e in c.imv_episodes:
                order(s + r.uniform(0.2, 1.0), "cxr", "ett")
            hour_of_day = (admit.hour + np.arange(n)) % 24
            on_imv = c.level == 4
            for i in np.nonzero((hour_of_day == 12) & on_imv)[0]:
                if r.random() < 0.7:
                    order(i + r.uniform(0, 1), "cxr", "daily_imv")
            starts = lambda m: [i for i in range(n) if m[i] and (i == 0 or not m[i - 1])]
            for i in starts(c.central_line) + starts(c.dialysis_line):
                order(i + r.uniform(0.3, 1.5), "cxr", "line")
            for cat, placed, removed in tubes.get(hid, []):
                h = (placed - admit).total_seconds() / 3600
                if cat in ("chest_tube", "pigtail"):
                    order(h + r.uniform(0.3, 1.5), "cxr", "chest_tube")
                    end = (discharge if pd.isna(removed) else removed)
                    order((end - admit).total_seconds() / 3600 + r.uniform(1, 4), "cxr", "chest_tube_out")
                elif cat == "dobhoff" or r.random() < 0.3:
                    order(h + r.uniform(0.3, 2.0), "kub", "feeding_tube")

            kidney_bad = lambda h: c.aki[min(int(h), n - 1)] > 0.5
            if c.infection_recognized_h is not None:
                h0 = c.infection_recognized_h + r.uniform(1, 12)
                src = c.infection_source if c.infected else None
                if src == "abdominal" or (src is None and r.random() < 0.15):
                    if r.random() < 0.85:
                        order(h0, "ct_ap" if kidney_bad(h0) else "ct_ap_c", "infection")
                elif src == "pulmonary" and r.random() < 0.3:
                    order(h0, "ct_chest" if kidney_bad(h0) else "ct_chest_c", "infection")
            if c.delirium.any() and r.random() < 0.25:
                order(float(np.argmax(c.delirium)) + r.uniform(1, 12), "ct_head", "delirium")
            gcs = c.gcs_eye + c.gcs_verbal + c.gcs_motor
            coma = (gcs <= 8) & (c.sed_depth < 0.2)
            if coma.any() and r.random() < 0.6:
                order(float(np.argmax(coma)) + r.uniform(0.5, 6), "ct_head", "coma")
            if c.shock.max() > 0.8 and r.random() < 0.3:
                order(float(np.argmax(c.shock > 0.8)) + r.uniform(24, 48), "us_ruq", "shock_liver")
            elif r.random() < 0.04:
                order(r.uniform(2, max(c.los_h * 0.5, 3)), "us_ruq", "lfts")
            if (c.aki > 0.4).any() and r.random() < 0.4:
                order(float(np.argmax(c.aki > 0.4)) + r.uniform(2, 24), "us_renal", "aki")
            if (c.ne_eq > 0).any() and (c.shock_type == "cardiogenic" or r.random() < 0.4):
                order(float(np.argmax(c.ne_eq > 0)) + r.uniform(1, 24), "echo", "shock")
            if c.los_h > 120 and crit.any() and r.random() < 0.08:
                order(r.uniform(72, c.los_h - 1), "le_doppler", "dvt")
            if crit.any() and r.random() < 0.04:
                h = r.uniform(2, max(c.los_h - 2, 3))
                if not kidney_bad(h):
                    order(h, "cta_chest", "pe")

            for k, (h, key, why) in enumerate(sorted(orders)):
                name, modality, region, contrast, portable_ok = STUDIES[key]
                t_order = at(admit, h)
                loc = locs.at(hid, t_order)
                sick = crit[min(int(h), n - 1)] or loc == "icu"
                portable = portable_ok and (sick or (key == "cxr" and r.random() < 0.5))
                if key == "cxr" and not portable:
                    name = "XR CHEST 1 VIEW"
                lag_cap = r.uniform(0.15, 1.5) if portable else r.uniform(0.5, 4.0)
                t_cap = t_order + pd.Timedelta(float(lag_cap), unit="h")
                if t_cap >= discharge:
                    continue
                lag_rep = r.uniform(0.2, 6.0) if modality == "xr" else r.uniform(0.4, 3.0)
                if modality == "us" and region == "heart":
                    lag_rep = r.uniform(2, 20)
                t_rep = t_cap + pd.Timedelta(float(lag_rep), unit="h")
                rows.append({
                    "patient_id": pid,
                    "radiology_accession": f"ACC{int(stable_id(hid, k, key, n=12), 16) % 10**10:010d}",
                    "radiology_order_dttm": t_order,
                    "radiology_capture_dttm": t_cap,
                    "radiology_report_dttm": t_rep,
                    "radiology_order_name": name,
                    "radiology_modality_category": modality,
                    "radiology_region_category": region,
                    "iv_contrast_category": contrast,
                    "radiology_location_category": "portable" if portable else "radiology",
                    "radiology_report_text": _report(c, key, why, h, r, tubes.get(hid, []), t_cap),
                })
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        for col in ("radiology_order_dttm", "radiology_capture_dttm", "radiology_report_dttm"):
            df[col] = pd.to_datetime(df[col], utc=True)
        return df


def _report(c, key, why, h, r, tubes, t_cap) -> str:
    """Short synthetic impression consistent with the course at hour h."""
    i = min(int(h), c.n_hours - 1)
    out = []
    if key in ("cxr", "cxr_2v"):
        if c.level[i] == 4:
            out.append(f"Endotracheal tube tip {r.uniform(3, 6):.1f} cm above the carina."
                       if not c.trach[i] else "Tracheostomy tube in place.")
        if c.central_line[i] or c.dialysis_line[i]:
            out.append("Right internal jugular catheter tip in the lower SVC.")
        in_chest = [t for t in tubes if t[0] in ("chest_tube", "pigtail") and t[1] <= t_cap
                    and (pd.isna(t[2]) or t_cap <= t[2])]
        if in_chest:
            out.append("Pleural catheter in place.")
        if why == "chest_tube_out":
            out.append("Chest tube removed. No pneumothorax.")
        resp = c.resp[i]
        if resp > 0.65:
            out.append("Diffuse bilateral airspace opacities.")
        elif resp > 0.4:
            out.append("Patchy bibasilar opacities, atelectasis versus pneumonia.")
        elif c.infected and c.infection_source == "pulmonary":
            out.append("Focal consolidation in the right lower lobe.")
        else:
            out.append("No focal consolidation.")
        if not any("pneumothorax" in s for s in out):
            out.append("No pneumothorax.")
    elif key == "kub":
        out.append("Enteric tube tip projects over the " +
                   ("duodenum." if any(t[0] == "dobhoff" for t in tubes) and r.random() < 0.7 else "stomach."))
    elif key in ("ct_ap", "ct_ap_c"):
        out.append("Rim-enhancing fluid collection, consistent with abscess." if c.infected and
                   c.infection_source == "abdominal" else "No acute intra-abdominal process.")
    elif key in ("ct_chest", "ct_chest_c"):
        out.append("Dense consolidation with air bronchograms." if c.infected else "No consolidation.")
    elif key == "cta_chest":
        out.append("No pulmonary embolism." if r.random() < 0.85 else "Segmental pulmonary embolism.")
    elif key == "ct_head":
        out.append("No acute intracranial abnormality.")
    elif key == "us_ruq":
        out.append("Hepatic congestion. No biliary dilatation." if c.shock.max() > 0.8
                   else "Normal liver and gallbladder.")
    elif key == "us_renal":
        out.append("Echogenic kidneys without hydronephrosis.")
    elif key == "echo":
        ef = r.uniform(15, 35) if c.shock_type == "cardiogenic" else r.uniform(50, 70)
        out.append(f"Left ventricular ejection fraction {ef:.0f}%.")
    elif key == "le_doppler":
        out.append("No deep venous thrombosis." if r.random() < 0.85 else "Acute DVT in the left popliteal vein.")
    return "IMPRESSION: " + " ".join(out) + " [synthetic report]"
