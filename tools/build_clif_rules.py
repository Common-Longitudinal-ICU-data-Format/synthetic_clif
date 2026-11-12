"""Build synthetic_clif/validation/clif_rules.json from clifpy's validation config.

The integrity checks in ``synthetic_clif.validation`` follow the rules the CLIF
consortium's own validator (clifpy) applies to real site data: relational
integrity, composite keys, chronological order, overlapping periods, numeric
outlier ranges, lab and vital units, and medication dose-unit rules. This tool
copies those rules from an installed clifpy so the package needs no clifpy at
runtime.

Usage:
  pip install clifpy
  python tools/build_clif_rules.py --out synthetic_clif/validation/clif_rules.json
"""

import argparse
import json
import os

import yaml

_RULE_KEYS = [
    "relational_integrity",
    "composite_keys",
    "chronological_order",
    "overlapping_periods",
    "medication_dose_unit_rules",
    "conditional_requirements",
    "cross_table_conditional_requirements",
]
_LAB_KEYS = ["lab_reference_units", "allowed_unit_variants", "lab_outlier_ranges"]
_VITAL_KEYS = ["vital_units", "vital_ranges"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clifpy-schemas", default=None,
                    help="clifpy/schemas directory (default: from the installed clifpy)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clifpy_version = None
    if a.clifpy_schemas is None:
        import clifpy
        from importlib.metadata import version

        a.clifpy_schemas = os.path.join(os.path.dirname(clifpy.__file__), "schemas")
        clifpy_version = version("clifpy")

    rules = yaml.safe_load(open(os.path.join(a.clifpy_schemas, "validation_rules.yaml")))
    outliers = yaml.safe_load(open(os.path.join(a.clifpy_schemas, "outlier_config.yaml")))

    out = {
        "source": f"clifpy {clifpy_version or '(path)'} schemas/validation_rules.yaml, "
                  "outlier_config.yaml and per-version labs/vitals schemas",
        "rules": {k: rules.get(k) for k in _RULE_KEYS},
        "outlier_ranges": outliers["tables"],
        "versions": {},
    }
    for ver in ("2.1", "3.0"):
        labs = yaml.safe_load(open(os.path.join(a.clifpy_schemas, ver, "labs_schema.yaml")))
        vit = yaml.safe_load(open(os.path.join(a.clifpy_schemas, ver, "vitals_schema.yaml")))
        out["versions"][ver] = {
            **{k: labs.get(k) for k in _LAB_KEYS},
            **{k: vit.get(k) for k in _VITAL_KEYS},
        }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")


if __name__ == "__main__":
    main()
