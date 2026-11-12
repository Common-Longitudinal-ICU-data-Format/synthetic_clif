"""Lightweight CLIF 3.0.0 conformance checks against the vendored schema.

These checks need no external package and cover every 3.0 table, including
the ones that clifpy does not yet ship table classes for. They test structure
and vocabularies only (not clinical plausibility):

* ``missing_columns`` / ``extra_columns``: column set differs from the schema
* ``null_required``: a required identifier or timestamp column has nulls
* ``invalid_categorical``: a value outside the column's permissible values
* ``group_mismatch``: a ``*_group`` value disagrees with the mCIDE mapping
* ``wrong_dtype``: a column's dtype does not match its CLIF type (DATETIME must be
  timezone-aware UTC, DATE tz-naive, DOUBLE float, INT integer, VARCHAR string)
"""

from __future__ import annotations

import pandas as pd

from synthetic_clif.clif3 import schema

# Required columns allowed to be null (mCIDE "allow_missing" or not simulated)
_NULLABLE_REQUIRED = {
    "patient": {"death_dttm", "race_category", "ethnicity_category", "language_category"},
    "hospitalization": {"discharge_dttm", "discharge_category"},
    "respiratory_support": None,  # settings are device-dependent
    "adt": {"location_type", "out_dttm"},
    "mcs": {"config_category"},  # IABP has no configuration category
    "airway": set(),
    "line": set(),
}


_EXPECTED_DTYPE = {
    "DATETIME": "datetime64[ns, UTC]",
    "DATE": "datetime64 (tz-naive)",
    "DOUBLE": "float",
    "FLOAT": "float",
    "INT": "integer",
    "BIGINT": "integer",
    "VARCHAR": "string",
}


def _dtype_ok(data_type: str, s: pd.Series) -> bool:
    t = s.dtype
    if data_type == "DATETIME":
        return pd.api.types.is_datetime64_any_dtype(t) and str(getattr(s.dt, "tz", None)) == "UTC"
    if data_type == "DATE":
        return pd.api.types.is_datetime64_any_dtype(t) and getattr(s.dt, "tz", None) is None
    if data_type in ("DOUBLE", "FLOAT"):
        return pd.api.types.is_float_dtype(t)
    if data_type in ("INT", "BIGINT"):
        return pd.api.types.is_integer_dtype(t)
    if data_type == "VARCHAR":
        if t == object:  # fine if every value is a string
            return bool(s.dropna().map(lambda v: isinstance(v, str)).all())
        return pd.api.types.is_string_dtype(t)
    return True


def _key_like(col: str) -> bool:
    return col.endswith("_id") or col.endswith("_dttm")


def validate_table(name: str, df: pd.DataFrame) -> list[dict]:
    errors: list[dict] = []
    tbl = schema.table(name)
    cols = [c["name"] for c in tbl["columns"]]

    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    if missing:
        errors.append({"type": "missing_columns", "table": name, "columns": missing})
    if extra:
        errors.append({"type": "extra_columns", "table": name, "columns": extra})

    nullable = _NULLABLE_REQUIRED.get(name, set())
    for c in tbl["columns"]:
        n = c["name"]
        if n not in df.columns:
            continue
        if c["required"] and nullable is not None and n not in nullable and _key_like(n):
            if df[n].isna().any():
                errors.append(
                    {"type": "null_required", "table": name, "column": n,
                     "n_null": int(df[n].isna().sum())}
                )
        pv = c.get("permissible_values")
        if pv is not None:
            allowed = set(map(str, pv))
            vals = df[n].dropna()
            if c["data_type"].upper() in ("INT", "BIGINT"):
                vals = vals.astype("int64")
            bad = sorted(set(map(str, vals.unique())) - allowed)
            if bad:
                errors.append(
                    {"type": "invalid_categorical", "table": name, "column": n, "values": bad[:20]}
                )
        dt = c["data_type"].upper()
        if len(df) and not _dtype_ok(dt, df[n]):
            errors.append({"type": "wrong_dtype", "table": name, "column": n,
                           "expected": _EXPECTED_DTYPE.get(dt, dt), "actual": str(df[n].dtype)})

    for key, mapping in tbl.items():
        if not key.endswith("_to_group_mapping"):
            continue
        cat = key[: -len("_to_group_mapping")]
        grp = cat.replace("_category", "_group")
        if cat in df.columns and grp in df.columns:
            flat = {k: (v[0] if isinstance(v, list) and v else v) for k, v in mapping.items()}
            sub = df[[cat, grp]].dropna()
            expected = sub[cat].map(flat)
            bad = sub[expected.notna() & (expected != sub[grp])]
            if len(bad):
                errors.append({"type": "group_mismatch", "table": name, "column": grp,
                               "examples": bad.drop_duplicates().head(5).values.tolist()})
    return errors


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
    """Validate each 3.0 table; returns {table: [errors]} (empty list = valid).

    Tables whose name is not a CLIF 3.0 table are reported as ``unknown_table``.
    """
    known = set(schema.table_names())
    out: dict[str, list[dict]] = {}
    for name, df in tables.items():
        if name not in known:
            out[name] = [{"type": "unknown_table", "table": name}]
        else:
            out[name] = validate_table(name, df)
    return out
