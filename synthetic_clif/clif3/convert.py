"""Convert a generated CLIF 2.1 dataset into CLIF 3.0.0 tables.

The generators produce CLIF 2.1 tables. ``convert_to_clif3`` turns that set
into CLIF 3.0.0:

* every ``*_category`` / ``*_group`` value is moved to the 3.0 conventions
  (lowercase snake_case plus the official renames, see ``vocab.py``);
* group columns are re-derived from the 3.0 category -> group mappings;
* renamed or split tables are restructured: ``ecmo_mcs`` -> ``mcs`` (long
  format), ``intake_output`` -> ``input`` + ``output``, ``key_icu_orders`` ->
  ``consult_orders``;
* renamed columns are moved (e.g. ``measure_*`` -> ``measurement_*``,
  ``volume_units`` -> ``volume_unit``) and new 3.0 columns are filled where the
  simulation implies a value (ADT ``room_id``/``bed_id``, trial
  ``participant_id``/``enrollment_dttm``, culture ``order_id``, ...);
* new 3.0 tables that follow from the simulated course are derived:
  ``airway`` (from invasive ventilation), ``line`` (from vasopressors, CRRT,
  PA-catheter data and arterial access), ``ed_encounter`` (from ED ADT
  segments) and ``misc_icu_orders`` (extubation orders);
* every table is conformed to the 3.0 column list and order.

The conversion is deterministic for a given ``seed``.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from synthetic_clif.clif3 import schema, vocab
from synthetic_clif.config import lab_catalog

# 2.1 tables whose names are unchanged in 3.0
_SAME_NAME = {
    "patient",
    "hospitalization",
    "adt",
    "vitals",
    "labs",
    "respiratory_support",
    "medication_admin_continuous",
    "medication_admin_intermittent",
    "microbiology_culture",
    "microbiology_susceptibility",
    "microbiology_nonculture",
    "patient_assessments",
    "patient_procedures",
    "hospital_diagnosis",
    "code_status",
    "position",
    "crrt_therapy",
    "clinical_trial",
    "invasive_hemodynamics",
    "medication_orders",
    "patient_diagnosis",
    "place_based_index",
    "provider",
    "therapy_details",
    "transfusion",
}

VASOPRESSORS = {"norepinephrine", "epinephrine", "vasopressin", "phenylephrine", "dopamine"}


def _hid(*parts) -> str:
    """Short stable hex id from the given parts."""
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()
    return h[:10]


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


class CLIF3Converter:
    """Stateful converter (holds the RNG and the 2.1 tables)."""

    def __init__(self, tables: dict[str, pd.DataFrame], seed: int = 42):
        self.src = tables
        self.rng = np.random.default_rng(seed + 3_000)

    # ------------------------------------------------------------------ API
    def convert(self, derive_new_tables: bool = True) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for name, df in self.src.items():
            if name in _SAME_NAME:
                fn = getattr(self, f"_t_{name}", None)
                # empty tables can lack columns; _conform adds them
                out[name] = fn(df.copy()) if fn and not df.empty else df.copy()
            elif name == "ecmo_mcs":
                out["mcs"] = self._t_mcs(df) if not df.empty else pd.DataFrame()
            elif name == "intake_output":
                if df.empty:
                    out["input"], out["output"] = pd.DataFrame(), pd.DataFrame()
                else:
                    out["input"], out["output"] = self._t_intake_output(df)
            elif name == "key_icu_orders":
                out["consult_orders"] = (self._t_consult_orders(df) if not df.empty
                                         else pd.DataFrame())
            # anything else has no 3.0 counterpart and is dropped

        if derive_new_tables:
            airway = self._derive_airway()
            if airway is not None:
                out["airway"] = airway
                out["misc_icu_orders"] = self._derive_extubation_orders(airway)
            line = self._derive_line()
            if line is not None:
                out["line"] = line
            ed = self._derive_ed_encounter()
            if ed is not None:
                out["ed_encounter"] = ed

        return {name: self._conform(name, df) for name, df in out.items()}

    # ------------------------------------------------- generic conformance
    def _conform(self, name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Map vocabularies, derive groups, add/reorder/coerce columns."""
        df = df.copy()
        tbl = schema.table(name)

        # 1. vocabularies for every column with a permissible list
        for col in tbl["columns"]:
            if col["name"] in df.columns and col.get("permissible_values") and col[
                "data_type"
            ].upper() == "VARCHAR":
                df[col["name"]] = vocab.map_column(df, name, col["name"])

        # 2. category -> group mappings
        for key, mapping in tbl.items():
            if not key.endswith("_to_group_mapping"):
                continue
            cat_col = key[: -len("_to_group_mapping")]
            grp_col = cat_col.replace("_category", "_group")
            if cat_col in df.columns:
                # some mCIDE mappings list several groups; the first is primary
                flat = {k: (v[0] if isinstance(v, list) and v else v) for k, v in mapping.items()}
                df[grp_col] = df[cat_col].map(flat)

        # 3. add missing columns, drop columns that 3.0 does not define, order
        cols = [c["name"] for c in tbl["columns"]]
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[cols]

        # 4. dtypes
        for col in tbl["columns"]:
            c, dt = col["name"], col["data_type"].upper()
            try:
                if dt == "DATETIME":
                    df[c] = _utc(df[c])
                elif dt == "DATE":
                    s = pd.to_datetime(df[c])
                    if getattr(s.dt, "tz", None) is not None:
                        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
                    df[c] = s.dt.normalize()  # midnight, tz-naive (clifpy: castable to DATE)
                elif dt in ("DOUBLE", "FLOAT"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
                elif dt in ("INT", "BIGINT"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")
                elif dt == "VARCHAR":
                    df[c] = df[c].astype("string")
            except (TypeError, ValueError):
                pass
        return df.reset_index(drop=True)

    # ------------------------------------------------------ per-table fixes
    def _t_patient(self, df):
        if "language_category" in df.columns:
            lang = df["language_category"].astype("string")
            non_english = lang.notna() & ~lang.str.lower().isin(
                ["english", "unknown", "unknown or na"]
            )
            # most non-English-speaking inpatients have an interpreter documented
            interp = non_english & (self.rng.random(len(df)) < 0.85)
            df["language_interpreter"] = np.where(lang.isna(), pd.NA, interp.astype(int))
        return df

    def _t_adt(self, df):
        units = df["location_name"].fillna(df["location_category"]).astype(str)
        prefix = units.map(lambda s: "".join(w[0] for w in s.split() if w[:1].isalpha()).upper())
        room = self.rng.integers(1, 40, len(df))
        floor = self.rng.integers(2, 12, len(df))
        df["room_id"] = [f"{p}-{f}{r:02d}" for p, f, r in zip(prefix, floor, room)]
        is_icu = df["location_category"].astype(str).str.lower().eq("icu")
        bed = np.where(self.rng.random(len(df)) < 0.5, "A", "B")
        df["bed_id"] = np.where(is_icu, df["room_id"] + "-1", df["room_id"] + "-" + bed)
        return df

    def _t_vitals(self, df):
        # charting time == measurement time in the simulation
        df["measured_dttm"] = df["recorded_dttm"]
        return df

    def _t_labs(self, df):
        # 3.0 reports troponins in ng/L (2.1 synthetic data used ng/mL)
        trop = df["lab_category"].isin(["troponin_i", "troponin_t"]) & df[
            "reference_unit"
        ].astype(str).str.lower().eq("ng/ml")
        if trop.any():
            num = pd.to_numeric(df.loc[trop, "lab_value_numeric"], errors="coerce") * 1000
            df.loc[trop, "lab_value_numeric"] = num.round(1)
            df.loc[trop, "lab_value"] = num.round(1).astype(str)
            df.loc[trop, "reference_unit"] = "ng/L"
        # 3.0 mCIDE units and order categories (cardiac, inflammatory, poc,
        # mixed_venous_blood_gas are new); bicarbonate keeps bmp or blood_gas
        cat = df["lab_category"].astype(str)
        known = cat.isin(list(lab_catalog.LABS))
        df.loc[known, "reference_unit"] = cat[known].map(lambda c: lab_catalog.LABS[c].unit_30)
        order30 = cat.map(lambda c: lab_catalog.LABS[c].order_30 if c in lab_catalog.LABS else None)
        df["lab_order_category"] = order30.where(order30.notna(), df["lab_order_category"])
        df["lab_category"] = cat.replace(lab_catalog.V3_CATEGORY)
        # order time is required in 3.0; fill the few missing ones before collection
        miss = df["lab_order_dttm"].isna() & df["lab_collect_dttm"].notna()
        if miss.any():
            lead = pd.to_timedelta(self.rng.uniform(5, 60, int(miss.sum())), unit="m")
            df.loc[miss, "lab_order_dttm"] = _utc(df.loc[miss, "lab_collect_dttm"]) - lead
        # one result per (stay, collection time, analyte): 3.0 composite key
        df = df.drop_duplicates(["hospitalization_id", "lab_collect_dttm", "lab_category"])
        if "lab_loinc_code" in df.columns:
            df["loinc_version"] = np.where(df["lab_loinc_code"].notna(), "2.81", None)
        return df

    def _t_microbiology_culture(self, df):
        keys = df[["hospitalization_id", "order_dttm", "fluid_category"]].astype(str).agg("|".join, axis=1)
        df["order_id"] = keys.map(lambda k: "ORD-" + _hid("culture", k))
        return df

    def _t_microbiology_nonculture(self, df):
        if "result_units" in df.columns and "result_unit" not in df.columns:
            df = df.rename(columns={"result_units": "result_unit"})
        return df

    def _t_invasive_hemodynamics(self, df):
        df = df.rename(
            columns={
                "measure_name": "measurement_name",
                "measure_category": "measurement_category",
                "measure_value": "measurement_value",
            }
        )
        # 2.1 categories were coarse (pressure/flow/...); 3.0 uses the measure
        df["measurement_category"] = df["measurement_name"]
        df["measurement_category"] = vocab.map_column(df, "invasive_hemodynamics", "measurement_category")
        pv = set(schema.permissible_values("invasive_hemodynamics", "measurement_category"))
        return df[df["measurement_category"].isin(pv)]

    def _t_transfusion(self, df):
        df = df.rename(columns={"volume_units": "volume_unit"})
        df["component_category"] = df["component_name"]
        return df

    def _t_clinical_trial(self, df):
        hosp = self.src.get("hospitalization")
        if hosp is not None:
            df = df.merge(hosp[["hospitalization_id", "patient_id"]], on="hospitalization_id", how="left")
        df["participant_id"] = [
            f"{t}-P{_hid('participant', t, h)[:5].upper()}"
            for t, h in zip(df["trial_id"], df["hospitalization_id"])
        ]
        consent = _utc(df["consent_dttm"])
        rand = _utc(df["randomized_dttm"])
        frac = self.rng.uniform(0.2, 0.9, len(df))
        df["enrollment_dttm"] = consent + (rand - consent) * frac
        return df

    # ---------------------------------------------------- provider linkage
    def _assign_provider(self, df: pd.DataFrame, time_col: str, roles: tuple[str, ...],
                         prefix: str) -> pd.Series:
        """Provider on shift (by role) at each row's time; synthetic id if none."""
        fallback = df["hospitalization_id"].astype(str).map(lambda h: f"{prefix}-{_hid(prefix, h)[:4].upper()}")
        prov = self.src.get("provider")
        if prov is None or prov.empty or df.empty:
            return fallback
        p = prov[prov["provider_role_category"].astype(str).isin(roles)][
            ["hospitalization_id", "provider_id", "start_dttm", "stop_dttm"]
        ].copy()
        if p.empty:
            return fallback
        p["start_dttm"] = _utc(p["start_dttm"])
        p["stop_dttm"] = _utc(p["stop_dttm"])
        left = pd.DataFrame({"hospitalization_id": df["hospitalization_id"].astype(str).to_numpy(),
                             "t": _utc(df[time_col]).to_numpy(), "_row": np.arange(len(df))})
        left = left.dropna(subset=["t"]).sort_values("t")
        p["hospitalization_id"] = p["hospitalization_id"].astype(str)
        m = pd.merge_asof(left, p.sort_values("start_dttm"), left_on="t", right_on="start_dttm",
                          by="hospitalization_id", direction="backward")
        ok = m["provider_id"].notna() & (m["stop_dttm"].isna() | (m["t"] <= m["stop_dttm"]))
        # outside any documented shift: use the closest shift of that role so the
        # id still exists in the provider table
        near = pd.merge_asof(left, p.sort_values("start_dttm"), left_on="t", right_on="start_dttm",
                             by="hospitalization_id", direction="nearest")
        assigned = pd.Series(fallback.to_numpy(), index=np.arange(len(df)), dtype=object)
        has_near = near["provider_id"].notna()
        assigned.loc[near.loc[has_near, "_row"].to_numpy()] = near.loc[has_near, "provider_id"].to_numpy()
        assigned.loc[m.loc[ok, "_row"].to_numpy()] = m.loc[ok, "provider_id"].to_numpy()
        return pd.Series(assigned.to_numpy(), index=df.index)

    def _t_medication_admin_continuous(self, df):
        df["administering_provider_id"] = self._assign_provider(df, "admin_dttm", ("RN",), "RN")
        return df

    def _t_medication_admin_intermittent(self, df):
        df["administering_provider_id"] = self._assign_provider(df, "admin_dttm", ("RN",), "RN")
        return df

    def _t_medication_orders(self, df):
        tcol = next((c for c in ("ordered_dttm", "order_start_dttm", "order_dttm") if c in df.columns), None)
        if tcol is not None:
            df["ordering_provider_id"] = self._assign_provider(
                df, tcol, ("Attending", "Resident", "Fellow", "NP", "PA"), "MD")
        return df

    def _t_patient_assessments(self, df):
        df["documenting_provider_id"] = self._assign_provider(df, "recorded_dttm", ("RN",), "RN")
        return df

    # ---------------------------------------------------- restructured tables
    def _t_mcs(self, df: pd.DataFrame) -> pd.DataFrame:
        device_map = {
            "VV_ECMO": ("ecmo", "cardiohelp", "v_v", "VV ECMO"),
            "VA_ECMO": ("ecmo", "cardiohelp", "v_a", "VA ECMO"),
            "Impella_CP": ("lvad", "impella", "cp", "Impella CP"),
            "Impella_5.5": ("lvad", "impella", "5_5", "Impella 5.5"),
            "CentriMag_LV": ("lvad", "centrimag", "central", "CentriMag LVAD"),
            "IABP": ("iabp", "cardiosave", None, "IABP"),
        }
        rows = []
        for r in df.itertuples(index=False):
            support, device, config, label = device_map.get(
                r.device_category, ("ecmo", None, None, str(r.device_category))
            )
            base = {
                "hospitalization_id": r.hospitalization_id,
                "recorded_dttm": r.recorded_dttm,
                "support_name": label.split()[-1] if support == "ecmo" else label,
                "support_category": support,
                "device_name": r.device_name,
                "device_category": device,
                "config_name": label,
                "config_category": config,
            }

            def add(setting_name, setting_category, value):
                if value is None or pd.isna(value):
                    return
                rows.append(
                    {**base, "setting_name": setting_name, "setting_category": setting_category,
                     "setting_value": float(value)}
                )

            add("Pump Flow (L/min)", "flow", r.flow)
            add("Sweep Gas (L/min)", "sweep", r.sweep_set)
            add("FdO2", "fdo2", r.fdO2_set)
            cp = str(r.control_parameter_category or "").lower()
            if support == "ecmo":
                add("Pump Speed (RPM)", "rpm", r.control_parameter_value)
            elif device == "impella":
                # Impella performance level P1-P9 tracks delivered flow
                flow = r.flow if r.flow is not None and not pd.isna(r.flow) else 3.0
                add("P-Level", "performance", float(np.clip(round(2 + (flow - 2.0) * 2.0), 1, 9)))
            elif device == "centrimag":
                v = r.control_parameter_value
                if v is not None and not pd.isna(v):
                    # CentriMag runs at ~2,500-5,500 RPM
                    add("Pump Speed (RPM)", "rpm", v / 10 if v > 10_000 else v)
            elif support == "iabp":
                v = r.control_parameter_value
                add("Augmentation Ratio", "iabp_ratio", 1.0 if v is None or pd.isna(v) else v)
            elif cp:
                add(str(r.control_parameter_name), cp, r.control_parameter_value)
        return pd.DataFrame(rows)

    def _t_intake_output(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df["recorded_dttm"] = df["intake_dttm"]
        intake = df[df["in_out_flag"] == 1].copy()
        output = df[df["in_out_flag"] != 1].copy()

        input_map = {
            "Normal Saline": "iv_crystalloid",
            "Lactated Ringers": "iv_crystalloid",
            "D5W": "iv_crystalloid",
            "D5 1/2NS": "iv_crystalloid",
            "Plasma-Lyte": "iv_crystalloid",
            "Albumin 5%": "iv_colloid",
            "Osmolite": "enteral_nutrition",
            "Promote": "enteral_nutrition",
            "Nepro": "enteral_nutrition",
            "Jevity": "enteral_nutrition",
            "Glucerna": "enteral_nutrition",
            "PO Intake": "po_intake",
            "Packed RBCs": "blood_products",
            "Fresh Frozen Plasma": "blood_products",
            "Platelets": "blood_products",
            "Cryoprecipitate": "blood_products",
        }
        intake["input_name"] = intake["fluid_name"]
        intake["input_category"] = intake["fluid_name"].map(input_map).fillna("other")
        intake["input_volume"] = intake["amount"]

        # Urine output: indwelling catheter in the ICU, voided elsewhere
        output["output_name"] = "Urine Output"
        output["output_value"] = output["amount"]
        output["output_unit"] = "mL"
        output["output_category"] = "urethral"
        adt = self.src.get("adt")
        if adt is not None and len(output):
            icu = adt[adt["location_category"].astype(str).str.lower() == "icu"][
                ["hospitalization_id", "in_dttm", "out_dttm"]
            ]
            o = output[["hospitalization_id", "recorded_dttm"]].reset_index()
            m = o.merge(icu, on="hospitalization_id", how="inner")
            in_icu = m[(m["recorded_dttm"] >= m["in_dttm"]) & (m["recorded_dttm"] < m["out_dttm"])]["index"]
            output.loc[output.index.isin(in_icu), "output_category"] = "indwelling_urinary_catheter"
            output.loc[output.index.isin(in_icu), "output_name"] = "Urine Output (Foley)"
        return intake, output

    def _t_consult_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["order_status_category"] = df.get("order_status_name")
        return df

    # ----------------------------------------------------- derived 3.0 tables
    def _hosp_bounds(self) -> Optional[pd.DataFrame]:
        h = self.src.get("hospitalization")
        if h is None:
            return None
        h = h[["hospitalization_id", "admission_dttm", "discharge_dttm"]].copy()
        h["admission_dttm"] = _utc(h["admission_dttm"])
        h["discharge_dttm"] = _utc(h["discharge_dttm"])
        return h.set_index("hospitalization_id")

    def _derive_airway(self) -> Optional[pd.DataFrame]:
        rs = self.src.get("respiratory_support")
        if rs is None or rs.empty:
            return None
        bounds = self._hosp_bounds()
        pat = self.src.get("patient")
        hosp = self.src.get("hospitalization")
        sex = {}
        if pat is not None and hosp is not None:
            s = hosp[["hospitalization_id", "patient_id"]].merge(
                pat[["patient_id", "sex_category"]], on="patient_id", how="left"
            )
            sex = dict(zip(s["hospitalization_id"], s["sex_category"].astype(str).str.lower()))

        rs = rs[["hospitalization_id", "recorded_dttm", "device_category", "tracheostomy"]].dropna(
            subset=["device_category"]
        )
        rs = rs.assign(recorded_dttm=_utc(rs["recorded_dttm"])).sort_values(
            ["hospitalization_id", "recorded_dttm"]
        )
        rows = []
        for hid, g in rs.groupby("hospitalization_id", sort=False):
            dev = g["device_category"].astype(str).str.upper().eq("IMV").to_numpy()
            trach = g["tracheostomy"].fillna(0).astype(int).to_numpy() == 1
            t = g["recorded_dttm"].to_numpy()
            trach_time = pd.Timestamp(t[trach.argmax()]) if trach.any() else None
            if trach_time is not None and trach_time.tzinfo is None:
                trach_time = trach_time.tz_localize("UTC")
            discharge = bounds.loc[hid, "discharge_dttm"] if bounds is not None and hid in bounds.index else None
            female = sex.get(hid, "") == "female"

            # endotracheal tube episodes: runs of IMV rows before any tracheostomy
            i, n, k = 0, len(g), 0
            while i < n:
                if dev[i] and not trach[i]:
                    j = i
                    while j + 1 < n and dev[j + 1] and not trach[j + 1]:
                        j += 1
                    placed = pd.Timestamp(t[i]).tz_localize("UTC") if pd.Timestamp(t[i]).tzinfo is None else pd.Timestamp(t[i])
                    if j + 1 < n:
                        removed = pd.Timestamp(t[j + 1])
                        removed = removed.tz_localize("UTC") if removed.tzinfo is None else removed
                    else:
                        removed = discharge
                    k += 1
                    size = self.rng.choice([7.0, 7.5]) if female else self.rng.choice([7.5, 8.0])
                    rows.append(
                        {"hospitalization_id": hid, "airway_id": f"ETT-{_hid(hid, k)[:6]}",
                         "placed_dttm": placed, "removed_dttm": removed,
                         "airway_category": "ett", "airway_size": size, "is_cuffed": 1}
                    )
                    i = j + 1
                else:
                    i += 1
            if trach_time is not None:
                rows.append(
                    {"hospitalization_id": hid, "airway_id": f"TRACH-{_hid(hid, 'trach')[:6]}",
                     "placed_dttm": trach_time, "removed_dttm": pd.NaT,
                     "airway_category": "trach", "airway_size": float(self.rng.choice([6.0, 7.0, 8.0])),
                     "is_cuffed": 1}
                )
        if not rows:
            return None
        return pd.DataFrame(rows)

    def _derive_extubation_orders(self, airway: pd.DataFrame) -> pd.DataFrame:
        ett = airway[(airway["airway_category"] == "ett") & airway["removed_dttm"].notna()].copy()
        # a later tracheostomy is not an extubation
        trach = airway[airway["airway_category"] == "trach"].set_index("hospitalization_id")["placed_dttm"]
        ett = ett[~ett.apply(lambda r: r["hospitalization_id"] in trach.index
                             and trach.loc[r["hospitalization_id"]] == r["removed_dttm"], axis=1)]
        lead = pd.to_timedelta(self.rng.uniform(10, 90, len(ett)), unit="m")
        out = pd.DataFrame(
            {
                "hospitalization_id": ett["hospitalization_id"].to_numpy(),
                "order_dttm": (_utc(ett["removed_dttm"]) - lead).to_numpy(),
                "order_name": "Extubate Patient",
                "order_category": "extubation",
                "order_status_name": "Completed",
                "order_status_category": "completed",
            }
        )
        return out

    def _derive_line(self) -> Optional[pd.DataFrame]:
        bounds = self._hosp_bounds()
        if bounds is None:
            return None
        rows = []

        def clamp(ts, hid):
            a, d = bounds.loc[hid, "admission_dttm"], bounds.loc[hid, "discharge_dttm"]
            if pd.isna(ts):
                return ts
            if ts < a:
                return a
            if pd.notna(d) and ts > d:
                return d
            return ts

        def add(hid, cat, site, placed, removed, lumens=None):
            k = sum(1 for r in rows if r["hospitalization_id"] == hid and r["line_category"] == cat) + 1
            rows.append(
                {"hospitalization_id": hid, "line_id": f"{cat.upper()}-{_hid(hid, cat, k)[:6]}",
                 "placed_dttm": clamp(placed, hid), "removed_dttm": clamp(removed, hid),
                 "line_category": cat, "line_site": site, "lumen_count": lumens}
            )

        hours = lambda lo, hi: pd.Timedelta(float(self.rng.uniform(lo, hi)), unit="h")

        # Peripheral IV for every admission
        for hid, b in bounds.iterrows():
            add(hid, "piv", str(self.rng.choice(["right_brachial", "left_brachial", "other"], p=[0.3, 0.3, 0.4])),
                b["admission_dttm"], b["discharge_dttm"], 1)

        # Central venous catheter for vasopressors
        mac = self.src.get("medication_admin_continuous")
        if mac is not None and not mac.empty:
            vp = mac[mac["med_category"].astype(str).str.lower().isin(VASOPRESSORS)]
            for hid, g in vp.groupby("hospitalization_id"):
                if hid not in bounds.index:
                    continue
                t0, t1 = _utc(g["admin_dttm"]).min(), _utc(g["admin_dttm"]).max()
                site = str(self.rng.choice(
                    ["right_ij", "left_ij", "right_subclavian", "left_subclavian", "right_femoral", "left_femoral"],
                    p=[0.55, 0.12, 0.1, 0.05, 0.12, 0.06]))
                add(hid, "cvc", site, t0 - hours(0, 2), t1 + hours(12, 72), 3)
                add(hid, "art_line", str(self.rng.choice(["right_radial", "left_radial", "right_femoral"],
                                                          p=[0.55, 0.35, 0.1])),
                    t0 - hours(0, 3), t1 + hours(6, 48), 1)

        # Arterial line for invasive ventilation without vasopressors
        rs = self.src.get("respiratory_support")
        have_art = {r["hospitalization_id"] for r in rows if r["line_category"] == "art_line"}
        if rs is not None and not rs.empty:
            imv = rs[rs["device_category"].astype(str).str.upper() == "IMV"]
            for hid, g in imv.groupby("hospitalization_id"):
                if hid in have_art or hid not in bounds.index or self.rng.random() > 0.7:
                    continue
                t = _utc(g["recorded_dttm"])
                add(hid, "art_line", str(self.rng.choice(["right_radial", "left_radial"])),
                    t.min() + hours(0, 4), t.max() + hours(0, 24), 1)

        # Dialysis catheter for CRRT
        crrt = self.src.get("crrt_therapy")
        if crrt is not None and not crrt.empty:
            for hid, g in crrt.groupby("hospitalization_id"):
                if hid not in bounds.index:
                    continue
                t = _utc(g["recorded_dttm"])
                add(hid, "dialysis_catheter",
                    str(self.rng.choice(["right_ij", "left_ij", "right_femoral", "left_femoral"], p=[0.5, 0.1, 0.3, 0.1])),
                    t.min() - hours(1, 6), t.max() + hours(0, 48), 2)

        # Pulmonary artery catheter for invasive hemodynamics
        ih = self.src.get("invasive_hemodynamics")
        if ih is not None and not ih.empty:
            for hid, g in ih.groupby("hospitalization_id"):
                if hid not in bounds.index:
                    continue
                t = _utc(g["recorded_dttm"])
                add(hid, "swan_ganz", "right_ij", t.min() - hours(0.5, 3), t.max() + hours(1, 12), 4)

        if not rows:
            return None
        return pd.DataFrame(rows)

    def _derive_ed_encounter(self) -> Optional[pd.DataFrame]:
        adt = self.src.get("adt")
        if adt is None or adt.empty:
            return None
        ed = adt[adt["location_category"].astype(str).str.lower() == "ed"].copy()
        if ed.empty:
            return None
        ed["in_dttm"] = _utc(ed["in_dttm"])
        ed["out_dttm"] = _utc(ed["out_dttm"])
        ed = ed.sort_values(["hospitalization_id", "in_dttm"]).groupby("hospitalization_id").first().reset_index()

        hosp = self.src.get("hospitalization")
        adm_type = {}
        if hosp is not None and "admission_type_category" in hosp.columns:
            adm_type = dict(zip(hosp["hospitalization_id"], hosp["admission_type_category"].astype(str).str.lower()))

        n = len(ed)
        mode = self.rng.choice(
            ["ems_ground", "private_vehicle", "walk_in", "ems_air", "law_enforcement", "other"],
            p=[0.5, 0.27, 0.15, 0.03, 0.02, 0.03], size=n)
        transfer = ed["hospitalization_id"].map(adm_type).isin(["osh", "transfer"]).to_numpy()
        mode = np.where(transfer, "transfer_ambulance", mode)
        acuity = self.rng.choice(["level_1", "level_2", "level_3", "level_4"], p=[0.12, 0.55, 0.3, 0.03], size=n)

        los = (ed["out_dttm"] - ed["in_dttm"]).dt.total_seconds() / 60
        triage_min = np.minimum(self.rng.uniform(2, 20, n), np.maximum(los.to_numpy() * 0.2, 0))
        triage = ed["in_dttm"] + pd.to_timedelta(triage_min, unit="m")
        # admit decision usually 1-4 h before the patient physically leaves the ED
        boarding = np.minimum(self.rng.uniform(30, 240, n), np.maximum(los.to_numpy() - triage_min, 0) * 0.8)
        disposition = ed["out_dttm"] - pd.to_timedelta(boarding, unit="m")

        names = {"level_1": "1 - Resuscitation", "level_2": "2 - Emergent", "level_3": "3 - Urgent",
                 "level_4": "4 - Less Urgent"}
        return pd.DataFrame(
            {
                "hospitalization_id": ed["hospitalization_id"],
                "ed_encounter_id": [f"ED-{_hid('ed', h)[:8]}" for h in ed["hospitalization_id"]],
                "ed_arrival_dttm": ed["in_dttm"],
                "ed_departure_dttm": ed["out_dttm"],
                "arrival_mode_name": pd.Series(mode).str.replace("_", " ").str.title().to_numpy(),
                "arrival_mode_category": mode,
                "triage_system_name": "ESI",
                "triage_system_category": "esi",
                "triage_acuity_name": [names[a] for a in acuity],
                "triage_acuity_category": acuity,
                "triage_dttm": triage,
                "ed_disposition_name": "Admit",
                "ed_disposition_category": "admit",
                "ed_disposition_dttm": disposition,
                "ed_destination_name": "Admitted",
                "ed_destination_category": "still_admitted",
                "ed_observation_status": 0,
            }
        )


def convert_to_clif3(
    tables: dict[str, pd.DataFrame], seed: int = 42, derive_new_tables: bool = True
) -> dict[str, pd.DataFrame]:
    """Convert a dict of CLIF 2.1 tables into CLIF 3.0.0 tables."""
    return CLIF3Converter(tables, seed=seed).convert(derive_new_tables=derive_new_tables)
