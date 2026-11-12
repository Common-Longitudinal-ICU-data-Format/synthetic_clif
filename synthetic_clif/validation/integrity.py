"""Dataset-level integrity checks (see ``synthetic_clif.validation``)."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_RULES_PATH = Path(__file__).parent / "clif_rules.json"

Tables = dict[str, pd.DataFrame]
Issue = dict

CHECKS: dict[str, Callable[[Tables, str], list[Issue]]] = {}


def register_check(name: str):
    """Register ``fn(tables, clif_version) -> list[issue]`` as a named check."""

    def deco(fn):
        CHECKS[name] = fn
        return fn

    return deco


@lru_cache(maxsize=1)
def rules() -> dict:
    with open(_RULES_PATH) as fh:
        return json.load(fh)


def _norm(value) -> Optional[str]:
    """Compare category values across versions: 'High Flow NC' == 'high_flow_nc'."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().replace("&", "_and_").replace("/", "_or_")
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def _rule_value(table: str, col: str, value, version: str) -> Optional[str]:
    """A category value written in clifpy's (2.1-style) rules, in this version's spelling."""
    if version == "3.0":
        from synthetic_clif.clif3 import schema
        from synthetic_clif.clif3.vocab import to_v3_value

        pv = schema.permissible_values(table, col) if table in schema.table_names() else None
        value = to_v3_value(table, col, value, pv)
    return _norm(value)


def _unit(value) -> Optional[str]:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _issue(check, table, message, column=None, n=None, examples=None) -> Issue:
    out = {"check": check, "table": table, "message": message}
    if column is not None:
        out["column"] = column
    if n is not None:
        out["n"] = int(n)
    if examples is not None:
        out["examples"] = examples
    return out


def _examples(df: pd.DataFrame, cols: list[str], k: int = 3) -> list:
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(k).astype(str).values.tolist()


def _ts(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    else:
        out = pd.to_datetime(s, errors="coerce", utc=True)
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize("UTC")
    return out


# Table names that changed between 2.1 and 3.0, for rules written against 2.1
_V21_NAME = {"mcs": "ecmo_mcs", "input": "intake_output", "output": "intake_output",
             "consult_orders": "key_icu_orders"}


# ---------------------------------------------------------------------------
# Table-specific rules not in clifpy's config. Sibling generators can extend
# these dicts (or register a new check) rather than editing the checks.
# ---------------------------------------------------------------------------

#: composite keys beyond clifpy's ``composite_keys``
EXTRA_COMPOSITE_KEYS: dict[str, list[str]] = {
    "medication_orders": ["med_order_id"],
    "line": ["line_id"],
    "airway": ["airway_id"],
    "ed_encounter": ["ed_encounter_id"],
    "transfusion": ["hospitalization_id", "transfusion_start_dttm", "product_code"],
}

#: id column -> table that must contain it (beyond clifpy's relational_integrity)
EXTRA_FOREIGN_KEYS: dict[str, str] = {
    "med_order_id": "medication_orders",
}

#: (earlier, later, strict) column pairs checked in any table holding both
EXTRA_ORDER: list[tuple[str, str, bool]] = [
    ("admission_dttm", "discharge_dttm", True),
    ("in_dttm", "out_dttm", True),
    ("start_dttm", "stop_dttm", False),
    ("start_dttm", "end_dttm", False),
    ("placed_dttm", "removed_dttm", False),
    ("order_start_dttm", "order_end_dttm", False),
    ("transfusion_start_dttm", "transfusion_end_dttm", False),
    ("ed_arrival_dttm", "ed_departure_dttm", False),
    ("consent_dttm", "enrollment_dttm", False),
    ("enrollment_dttm", "withdrawal_dttm", False),
    ("lab_order_dttm", "lab_collect_dttm", False),
    ("lab_collect_dttm", "lab_result_dttm", False),
    ("order_dttm", "collect_dttm", False),
    ("collect_dttm", "result_dttm", False),
    ("measured_dttm", "recorded_dttm", False),
]

#: {table: {column: (min, max) or {category: (min, max)}}} numeric limits
#: beyond clifpy's outlier config. A per-category dict needs the table in
#: ``_CATEGORY_COLUMN``; categories it leaves out go unchecked.
_HEMO = {"cvp": (-10, 40), "pa_systolic": (0, 150), "pa_diastolic": (-10, 80),
         "pa_mean": (0, 100), "pcwp": (-5, 60), "co_thermodilution": (0.5, 20),
         "co_index": (0.3, 10), "svr": (100, 4000),
         # 2.1 names (measure_name; its measure_category is only a group)
         "cardiac_output": (0.5, 20), "cardiac_index": (0.3, 10),
         "pvr": (0, 1500), "svo2": (0, 100)}
EXTRA_RANGES: dict[str, dict] = {
    "transfusion": {"volume_transfused": (0, 5000)},
    "input": {"input_volume": (0, 20000)},
    "output": {"output_value": (0, 20000)},
    "intake_output": {"amount": (0, 20000)},
    "invasive_hemodynamics": {"measurement_value": _HEMO, "measure_value": _HEMO},
    "hospitalization": {"age_at_admission": (0, 120)},
}

#: {(table, column): (hours before admission, hours after discharge)} allowed.
#: Columns not listed must fall inside [admission, discharge]. ``None`` exempts
#: the column (history and future dates, e.g. a problem list).
STAY_WINDOW_ALLOWANCE: dict[tuple[str, str], Optional[tuple[float, float]]] = {
    # results can post after the patient leaves
    ("labs", "lab_result_dttm"): (0, 7 * 24),
    ("microbiology_culture", "result_dttm"): (0, 30 * 24),
    ("microbiology_nonculture", "result_dttm"): (0, 30 * 24),
    # billing and coding run after discharge
    ("patient_procedures", "procedure_billed_dttm"): (0, 30 * 24),
    # the ED visit precedes the inpatient admission
    ("ed_encounter", "ed_arrival_dttm"): (72, 0),
    ("ed_encounter", "triage_dttm"): (72, 0),
    ("ed_encounter", "ed_disposition_dttm"): (72, 0),
    ("ed_encounter", "ed_departure_dttm"): (72, 0),
    # problem-list and trial dates are not bound to one stay
    ("patient_diagnosis", "start_dttm"): None,
    ("patient_diagnosis", "end_dttm"): None,
    ("clinical_trial", "consent_dttm"): None,
    ("clinical_trial", "enrollment_dttm"): None,
    ("clinical_trial", "randomized_dttm"): None,
    ("clinical_trial", "withdrawal_dttm"): None,
    # orders and devices may outlive the stay's last charted minute
    ("medication_orders", "order_end_dttm"): None,
    ("line", "removed_dttm"): None,
    ("airway", "removed_dttm"): None,
    ("drain", "removed_dttm"): None,
}

# Tolerance for clocks that round or truncate (e.g. a vitals time vs admission)
_SLACK = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
@register_check("schema")
def check_schema(tables: Tables, version: str) -> list[Issue]:
    if version != "3.0":
        return []  # 2.1 conformance is checked by clifpy (tests/test_clifpy_validation.py)
    from synthetic_clif.clif3.validate import validate_tables

    out = []
    for table, errs in validate_tables(tables).items():
        for e in errs:
            e = dict(e)
            out.append(_issue("schema", table, f"{e.pop('type')}: {e}", column=e.get("column")))
    return out


def _composite_keys() -> dict[str, list[str]]:
    keys = {t: v["keys"] for t, v in (rules()["rules"]["composite_keys"] or {}).items()}
    keys.update(EXTRA_COMPOSITE_KEYS)
    return keys


@register_check("primary_key")
def check_primary_key(tables: Tables, version: str) -> list[Issue]:
    out = []
    for table, keys in _composite_keys().items():
        df = tables.get(table)
        if df is None or not len(df) or not set(keys) <= set(df.columns):
            continue
        dup = df[df.duplicated(subset=keys, keep=False)]
        if len(dup):
            out.append(_issue("primary_key", table, f"duplicate rows on {keys}", n=len(dup),
                              examples=_examples(dup.sort_values(keys), keys)))
    return out


def _foreign_keys() -> dict[str, str]:
    fks = {col: v["references_table"]
           for col, v in (rules()["rules"]["relational_integrity"] or {}).items()}
    fks.update(EXTRA_FOREIGN_KEYS)
    return fks


@register_check("foreign_key")
def check_foreign_key(tables: Tables, version: str) -> list[Issue]:
    out = []
    for col, ref in _foreign_keys().items():
        ref_df = tables.get(ref)
        if ref_df is None or col not in ref_df.columns:
            continue
        valid = set(ref_df[col].dropna().astype(str))
        for table, df in tables.items():
            if table == ref or col not in df.columns or not len(df):
                continue
            vals = df[col].dropna().astype(str)
            bad = vals[~vals.isin(valid)]
            if len(bad):
                out.append(_issue("foreign_key", table, f"{col} not found in {ref}", column=col,
                                  n=len(bad), examples=sorted(bad.unique())[:3]))
    hosp = tables.get("hospitalization")
    if hosp is not None and {"patient_id", "hospitalization_id"} <= set(hosp.columns):
        owner = hosp.drop_duplicates("hospitalization_id").set_index("hospitalization_id")["patient_id"]
        owner.index = owner.index.astype(str)
        owner = owner.astype(str)
        for table, df in tables.items():
            if table == "hospitalization" or not {"patient_id", "hospitalization_id"} <= set(df.columns):
                continue
            sub = df.dropna(subset=["patient_id", "hospitalization_id"])
            expect = sub["hospitalization_id"].astype(str).map(owner)
            bad = sub[expect.notna() & (expect != sub["patient_id"].astype(str))]
            if len(bad):
                out.append(_issue("foreign_key", table,
                                  "patient_id disagrees with the hospitalization's patient",
                                  column="patient_id", n=len(bad),
                                  examples=_examples(bad, ["patient_id", "hospitalization_id"])))
    return out


_GENERIC_PAIRS = (("start", "end"), ("start", "stop"), ("begin", "end"))


def _order_pairs(table: str, columns=()) -> list[tuple[str, str, bool]]:
    """clifpy's rules for the table, EXTRA_ORDER, and any ``*start*_dttm`` /
    ``*end*_dttm`` (or stop) pair among ``columns``, so new tables are covered."""
    pairs = {}
    for r in (rules()["rules"]["chronological_order"] or {}).get(table, []):
        pairs[(r["earlier"], r["later"])] = bool(r.get("strict", False))
    for a, b, strict in EXTRA_ORDER:
        pairs.setdefault((a, b), strict)
    cols = set(columns)
    for c in cols:
        if not c.endswith("_dttm"):
            continue
        for lo, hi in _GENERIC_PAIRS:
            if lo in c and c.replace(lo, hi) in cols:
                pairs.setdefault((c, c.replace(lo, hi)), False)
    return [(a, b, s) for (a, b), s in pairs.items()]


@register_check("chronology")
def check_chronology(tables: Tables, version: str) -> list[Issue]:
    out = []
    for table, df in tables.items():
        for a, b, strict in _order_pairs(table, df.columns):
            if a not in df.columns or b not in df.columns or not len(df):
                continue
            ta, tb = _ts(df[a]), _ts(df[b])
            both = ta.notna() & tb.notna()
            bad = both & ((tb <= ta) if strict else (tb < ta))
            if bad.any():
                rel = "after or at" if strict else "after"
                out.append(_issue("chronology", table, f"{a} {rel} {b}", column=b,
                                  n=bad.sum(), examples=_examples(df[bad], [a, b])))
    return out


def _overlaps(df: pd.DataFrame, entity: str, start: str, end: str) -> pd.DataFrame:
    d = df[[entity, start, end]].copy()
    d[start], d[end] = _ts(d[start]), _ts(d[end])
    d = d.dropna(subset=[start]).sort_values([entity, start])
    prev_end = d.groupby(entity)[end].shift()
    return d[prev_end.notna() & (d[start] < prev_end - _SLACK)]


@register_check("overlap")
def check_overlap(tables: Tables, version: str) -> list[Issue]:
    specs = [(t, v["entity_column"], v["start_column"], v["end_column"])
             for t, v in (rules()["rules"]["overlapping_periods"] or {}).items()]
    specs.append(("hospitalization", "patient_id", "admission_dttm", "discharge_dttm"))
    out = []
    for table, entity, start, end in specs:
        df = tables.get(table)
        if df is None or not {entity, start, end} <= set(df.columns):
            continue
        bad = _overlaps(df, entity, start, end)
        if len(bad):
            out.append(_issue("overlap", table, f"{start}-{end} periods overlap within {entity}",
                              n=len(bad), examples=_examples(bad, [entity, start, end])))
    return out


@register_check("stay_window")
def check_stay_window(tables: Tables, version: str) -> list[Issue]:
    hosp = tables.get("hospitalization")
    if hosp is None:
        return []
    win = hosp[["hospitalization_id", "patient_id", "admission_dttm", "discharge_dttm"]].copy()
    win["hospitalization_id"] = win["hospitalization_id"].astype(str)
    win["admission_dttm"], win["discharge_dttm"] = _ts(win["admission_dttm"]), _ts(win["discharge_dttm"])
    pat = tables.get("patient")
    if pat is not None and "death_dttm" in pat.columns:
        death = pat.set_index(pat["patient_id"].astype(str))["death_dttm"]
        win["death_dttm"] = _ts(win["patient_id"].astype(str).map(death))
    else:
        win["death_dttm"] = pd.NaT
    win = win.drop_duplicates("hospitalization_id").set_index("hospitalization_id")

    out = []
    for table, df in tables.items():
        if table == "hospitalization" or "hospitalization_id" not in df.columns or not len(df):
            continue
        hid = df["hospitalization_id"].astype(str)
        adm = hid.map(win["admission_dttm"])
        dis = hid.map(win["discharge_dttm"])
        dead = hid.map(win["death_dttm"])
        for col in [c for c in df.columns if c.endswith("_dttm")]:
            key = (table, col)
            if key in STAY_WINDOW_ALLOWANCE and STAY_WINDOW_ALLOWANCE[key] is None:
                continue
            before_h, after_h = STAY_WINDOW_ALLOWANCE.get(key) or (0, 0)
            t = _ts(df[col])
            lo = adm - pd.Timedelta(hours=before_h) - _SLACK
            hi = dis + pd.Timedelta(hours=after_h) + _SLACK
            early = t.notna() & adm.notna() & (t < lo)
            late = t.notna() & dis.notna() & (t > hi)
            after_death = t.notna() & dead.notna() & (t > dead + pd.Timedelta(hours=after_h) + _SLACK)
            for mask, what in ((early, "before admission"), (late, "after discharge"),
                               (after_death, "after the patient's death")):
                if mask.any():
                    out.append(_issue("stay_window", table, f"{col} {what}", column=col,
                                      n=mask.sum(),
                                      examples=_examples(df[mask], ["hospitalization_id", col])))
    return out


_CATEGORY_COLUMN = {
    "vitals": "vital_category",
    "labs": "lab_category",
    "patient_assessments": "assessment_category",
    "medication_admin_continuous": "med_category",
    "medication_admin_intermittent": "med_category",
    "ecmo_mcs": "device_category",
    "invasive_hemodynamics": ("measurement_category", "measure_name"),
}


def _cat_col(table: str, df: pd.DataFrame | None = None):
    """The column that holds a table's category, or None. Some tables name it
    differently by version, so the first candidate present in ``df`` wins."""
    c = _CATEGORY_COLUMN.get(table)
    if isinstance(c, tuple):
        if df is None:
            return c[0]
        return next((x for x in c if x in df.columns), None)
    return c


def _range_specs(table: str, version: str) -> dict:
    """{value column: {"__all__": (lo, hi)} or {norm category: (lo, hi)} or
    {norm category: {unit: (lo, hi)}}}, merged from clifpy and EXTRA_RANGES."""
    cfg = rules()["outlier_ranges"]
    raw = cfg.get(table) or cfg.get(_V21_NAME.get(table, "")) or {}
    specs: dict = {}
    for col, spec in raw.items():
        if "min" in spec and "max" in spec:
            specs[col] = {"__all__": (spec["min"], spec["max"])}
            continue
        by_cat = {}
        for cat, sub in spec.items():
            key = _rule_value(table, _cat_col(table) or "", cat, version)
            if "min" in sub and "max" in sub:
                by_cat[key] = (sub["min"], sub["max"])
            else:
                by_cat[key] = {_unit(u): (r["min"], r["max"]) for u, r in sub.items()}
        specs[col] = by_cat
    ver = rules()["versions"].get(version, {})
    if table == "labs":
        for cat, r in (ver.get("lab_outlier_ranges") or {}).items():
            specs.setdefault("lab_value_numeric", {})[_rule_value("labs", "lab_category", cat, version)] = (r["min"], r["max"])
    if table == "vitals":
        for cat, r in (ver.get("vital_ranges") or {}).items():
            cur = specs.setdefault("vital_value", {}).get(_norm(cat))
            lo, hi = r["min"], r["max"]
            if isinstance(cur, tuple):  # keep the wider of clifpy's two ranges
                lo, hi = min(lo, cur[0]), max(hi, cur[1])
            specs["vital_value"][_norm(cat)] = (lo, hi)
    for col, rng in EXTRA_RANGES.get(table, {}).items():
        if isinstance(rng, dict):
            specs[col] = {_rule_value(table, _cat_col(table) or "", c, version): r
                          for c, r in rng.items()}
        else:
            specs[col] = {"__all__": rng}
    return specs




def _bounds(table: str, df: pd.DataFrame, version: str):
    """Yield (column, values, lo, hi) with each row's range (NaN where none applies)."""
    for col, spec in _range_specs(table, version).items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if "__all__" in spec:
            lo = pd.Series(spec["__all__"][0], index=df.index, dtype=float)
            hi = pd.Series(spec["__all__"][1], index=df.index, dtype=float)
        else:
            cat_col = _cat_col(table, df)
            if cat_col is None or cat_col not in df.columns:
                continue
            cats = df[cat_col].map(_norm)
            lo = pd.Series(np.nan, index=df.index)
            hi = pd.Series(np.nan, index=df.index)
            units = df["med_dose_unit"].map(_unit) if "med_dose_unit" in df.columns else None
            for cat, r in spec.items():
                m = cats == cat
                if not m.any():
                    continue
                if isinstance(r, tuple):
                    lo[m], hi[m] = r
                elif units is not None:
                    for u, (a, b) in r.items():
                        mu = m & (units == u)
                        lo[mu], hi[mu] = a, b
        yield col, x, lo, hi


@register_check("value_range")
def check_value_range(tables: Tables, version: str) -> list[Issue]:
    out = []
    for table, df in tables.items():
        if not len(df):
            continue
        for col, x, lo, hi in _bounds(table, df, version):
            bad = x.notna() & lo.notna() & ((x < lo) | (x > hi))
            if bad.any():
                show = [c for c in (_cat_col(table, df), "med_dose_unit", col) if c and c in df.columns]
                out.append(_issue("value_range", table, f"{col} outside clifpy's outlier range",
                                  column=col, n=bad.sum(), examples=_examples(df[bad], show)))
    return out


def range_coverage(tables: Tables, clif_version: str = "3.0") -> dict[tuple[str, str], float]:
    """Share of non-null values in each checked column that have a range to check.

    A check whose rules match none of the data passes silently (issue #53);
    tests use this to make sure the range rules actually apply.
    """
    version = "2.1" if str(clif_version).startswith("2") else "3.0"
    out = {}
    for table, df in tables.items():
        if not len(df):
            continue
        for col, x, lo, _ in _bounds(table, df, version):
            n = int(x.notna().sum())
            if n:
                out[(table, col)] = float((x.notna() & lo.notna()).sum() / n)
    return out


_TIME_DENOM = re.compile(r"/\s*(?:min|minute|hr|hour|h|day|d)\b")


@register_check("units")
def check_units(tables: Tables, version: str) -> list[Issue]:
    out = []
    ver = rules()["versions"].get(version, {})
    labs = tables.get("labs")
    ref = ver.get("lab_reference_units") or {}
    variants = ver.get("allowed_unit_variants") or {}
    if labs is not None and len(labs) and {"lab_category", "reference_unit"} <= set(labs.columns):
        allowed = {}
        for cat, unit in ref.items():
            u = _unit(unit)
            allowed[_norm(cat)] = {u, *map(_unit, variants.get(u, []) or [])}
        cats = labs["lab_category"].map(_norm)
        units = labs["reference_unit"].map(_unit)
        ok = pd.Series(
            [c not in allowed or u is None or u in allowed[c] for c, u in zip(cats, units)],
            index=labs.index,
        )
        if (~ok).any():
            bad = labs[~ok]
            pairs = sorted(set(zip(bad["lab_category"].astype(str), bad["reference_unit"].astype(str))))
            out.append(_issue("units", "labs", "reference_unit is not the CLIF unit for its lab",
                              column="reference_unit", n=len(bad),
                              examples=[[c, u, sorted(allowed[_norm(c)])[0]] for c, u in pairs[:5]]))
    for table, r in (rules()["rules"]["medication_dose_unit_rules"] or {}).items():
        df = tables.get(table)
        if df is None or not len(df) or "med_dose_unit" not in df.columns:
            continue
        units = df["med_dose_unit"].dropna().astype(str).str.lower()
        rate = units.str.contains(_TIME_DENOM)
        bad = units[~rate] if r["expect"] == "per_time" else units[rate]
        if len(bad):
            out.append(_issue("units", table,
                              f"med_dose_unit should be {'a rate' if r['expect'] == 'per_time' else 'a dose'}",
                              column="med_dose_unit", n=len(bad), examples=sorted(bad.unique())[:5]))
    return out


@register_check("death")
def check_death(tables: Tables, version: str) -> list[Issue]:
    hosp, pat = tables.get("hospitalization"), tables.get("patient")
    if hosp is None or pat is None or "death_dttm" not in pat.columns:
        return []
    death = _ts(pat.set_index(pat["patient_id"].astype(str))["death_dttm"])
    h = hosp.copy()
    h["death_dttm"] = _ts(h["patient_id"].astype(str).map(death))
    expired = h["discharge_category"].map(_norm) == "expired"
    out = []
    missing = expired & h["death_dttm"].isna()
    if missing.any():
        out.append(_issue("death", "patient", "expired discharge with no death_dttm",
                          column="death_dttm", n=missing.sum(),
                          examples=_examples(h[missing], ["patient_id", "hospitalization_id"])))
    dis = _ts(h["discharge_dttm"])
    off = expired & h["death_dttm"].notna() & dis.notna() & (
        (h["death_dttm"] - dis).abs() > pd.Timedelta(hours=24))
    if off.any():
        out.append(_issue("death", "patient", "death_dttm more than 24 h from the expired discharge",
                          column="death_dttm", n=off.sum(),
                          examples=_examples(h[off], ["patient_id", "discharge_dttm", "death_dttm"])))
    return out


@register_check("never_charted")
def check_never_charted(tables: Tables, version: str, min_rows: int = 20) -> list[Issue]:
    """A setting clifpy requires for a device or mode that no row ever records.

    Synthetic data is allowed to miss values, but a required setting missing
    from every row (say, liter flow on high flow nasal cannula) means the
    generator never produces it.
    """
    out = []
    for table, reqs in (rules()["rules"]["conditional_requirements"] or {}).items():
        df = tables.get(table)
        if df is None or not len(df):
            continue
        for r in reqs:
            if r["when_column"] not in df.columns:
                continue
            when = {_rule_value(table, r["when_column"], v, version) for v in r["when_value"]}
            m = df[r["when_column"]].map(_norm).isin(when)
            if "and_column" in r:
                if r["and_column"] not in df.columns:
                    continue
                also = {_rule_value(table, r["and_column"], v, version) for v in r["and_value"]}
                m &= df[r["and_column"]].map(_norm).isin(also)
            if m.sum() < min_rows:
                continue
            for col in r["then_required"]:
                if col in df.columns and df.loc[m, col].isna().all():
                    out.append(_issue("never_charted", table,
                                      f"{col} is never recorded when {r['description']!r}",
                                      column=col, n=m.sum()))
    return out


# ---------------------------------------------------------------------------
def check_dataset(tables: Tables, clif_version: str = "3.0",
                  checks: Optional[list[str]] = None) -> list[Issue]:
    """Run the named checks (default: all) and return every issue found."""
    version = "2.1" if str(clif_version).startswith("2") else "3.0"
    issues: list[Issue] = []
    for name in checks or list(CHECKS):
        issues.extend(CHECKS[name](tables, version))
    return issues


def format_issues(issues: list[Issue]) -> str:
    return "\n".join(
        f"[{i['check']}] {i['table']}: {i['message']}"
        + (f" (n={i['n']})" if "n" in i else "")
        + (f" e.g. {i['examples']}" if i.get("examples") else "")
        for i in issues
    )
