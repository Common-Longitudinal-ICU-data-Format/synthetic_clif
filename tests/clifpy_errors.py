"""Which clifpy validation errors mean the data breaks the CLIF schema.

clifpy reports two kinds of finding on each table's ``errors``:

* schema errors (``category == "schema"``): missing required columns, invalid
  categorical values, invalid units, non-castable data types;
* data-quality warnings (``category == "data_quality"``): missingness, mCIDE
  coverage gaps, distribution shift, conditional requirements. Synthetic data
  is meant to have some of these.

Issue #53: the old test matched machine-style type keys ("missing_columns")
that clifpy no longer emits, so it matched nothing and every structural
assertion passed. Matching on clifpy's ``category`` field, plus its current
Title Case type names for releases without that field, fixes the filter, and
``tests/test_clifpy_validation.py::test_filter_catches_injected_errors``
fails if a future clifpy changes the strings again.
"""

SCHEMA_ERROR_TYPES = {
    "Missing Columns",
    "Missing Required Columns",
    "Invalid Categorical Values",
    "Invalid Units",
    "Category Group Mapping",
}

DTYPE_MISMATCH = "Data Type Mismatch"
MISSING_COLUMN_TYPES = {"Missing Columns", "Missing Required Columns"}

_TEMPORAL = ("TIMESTAMP", "DATE", "DATETIME")
_NUMERIC = ("INT", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL")


def _family(sql_type: str):
    t = str(sql_type).upper()
    if t.startswith(_TEMPORAL):
        return "temporal"
    if t.startswith(_NUMERIC):
        return "numeric"
    return t


def structural_errors(errors: list[dict], columns=None) -> list[dict]:
    """clifpy errors that are schema violations (not data-quality warnings).

    ``columns``: the table's actual columns. clifpy 2.x reports a mixed-case
    required column (ecmo_mcs ``fdO2``) as missing even when it is present;
    a "missing" column that the table has is dropped as a false positive.
    """
    present = set(columns) if columns is not None else set()
    out = []
    for e in errors:
        etype = e.get("type", "")
        details = e.get("details") or {}
        if etype == DTYPE_MISMATCH:
            # clifpy calls any type "castable" by type, so a column of text
            # where numbers belong counts too. Only a cast within one family
            # (TIMESTAMP stored for DATE, BIGINT for DOUBLE) is harmless.
            same = _family(details.get("expected")) == _family(details.get("actual"))
            if not (details.get("castable", False) and same):
                out.append(e)
        elif etype in MISSING_COLUMN_TYPES and details.get("column") in present:
            continue
        elif e.get("category") == "schema" or etype in SCHEMA_ERROR_TYPES:
            out.append(e)
    return out


def of_type(errors: list[dict], *types: str) -> list[dict]:
    return [e for e in errors if e.get("type") in types]
