"""CLIF 3.0.0 table schemas, vendored as JSON.

``clif3_schema.json`` is built by ``tools/build_clif3_schema.py`` from the
official CLIF 3.0 sources (the ``3.0`` branch of
Common-Longitudinal-ICU-data-Format/CLIF: MySQL DDL + mCIDE CSVs, as packaged
in clifpy's 3.0 YAML schemas). It holds, per table, the column order, data
types, required flags, permissible values and category -> group mappings.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_SCHEMA_PATH = Path(__file__).parent / "clif3_schema.json"

# CLIF 3.0 data types -> pandas dtypes used when coercing output columns
_PANDAS_DTYPE = {
    "VARCHAR": "string",
    "DOUBLE": "float64",
    "FLOAT": "float64",
    "INT": "Int64",
    "BIGINT": "Int64",
    "DATETIME": "datetime64[ns, UTC]",
    "DATE": "datetime64[ns]",
}


@lru_cache(maxsize=1)
def load_schema() -> dict:
    """Return the full vendored CLIF 3.0 schema document."""
    with open(_SCHEMA_PATH) as fh:
        return json.load(fh)


def table_names() -> list[str]:
    return list(load_schema()["tables"].keys())


def table(name: str) -> dict:
    return load_schema()["tables"][name]


def columns(name: str) -> list[str]:
    return [c["name"] for c in table(name)["columns"]]


def required_columns(name: str) -> list[str]:
    return [c["name"] for c in table(name)["columns"] if c["required"]]


def column(name: str, col: str) -> Optional[dict]:
    for c in table(name)["columns"]:
        if c["name"] == col:
            return c
    return None


def permissible_values(name: str, col: str) -> Optional[list]:
    c = column(name, col)
    return None if c is None else c.get("permissible_values")


def category_to_group(name: str, category_col: str) -> Optional[dict]:
    """Return the ``<x>_category -> <x>_group`` mapping for a table, if any."""
    return table(name).get(f"{category_col}_to_group_mapping")


def pandas_dtype(data_type: str) -> str:
    return _PANDAS_DTYPE.get(data_type.upper(), "object")
