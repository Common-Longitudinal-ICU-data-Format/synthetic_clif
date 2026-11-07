"""Build synthetic_clif/clif3/clif3_schema.json from the official CLIF 3.0 sources.

Inputs:
  --clifpy-schemas  path to clifpy/schemas/3.0 (YAML; column lists match the
                    CLIF 3.0 MySQL DDL on the CLIF repo's `3.0` branch)
  --mcide           path to the CLIF repo's mCIDE directory checked out at `3.0`

Usage:
  python tools/build_clif3_schema.py --clifpy-schemas ../clifpy/clifpy/schemas/3.0 \
      --mcide ../CLIF/mCIDE --out synthetic_clif/clif3/clif3_schema.json
"""

import argparse
import glob
import json
import os

import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clifpy-schemas", required=True)
    ap.add_argument("--mcide", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-note", default="")
    a = ap.parse_args()

    tables = {}
    for f in sorted(glob.glob(os.path.join(a.clifpy_schemas, "*_schema.yaml"))):
        s = yaml.safe_load(open(f))
        cols = []
        for c in s["columns"]:
            col = {
                "name": c["name"],
                "data_type": c["data_type"],
                "required": bool(c.get("required", False)),
            }
            if c.get("permissible_values") is not None:
                col["permissible_values"] = c["permissible_values"]
            cols.append(col)
        entry = {"columns": cols}
        for k, v in s.items():
            if k.endswith("_to_group_mapping"):
                entry[k] = v
        tables[s["table_name"]] = entry

    # category -> group mappings that live only in mCIDE CSVs (not in the YAML)
    nc_csv = os.path.join(
        a.mcide, "microbiology_nonculture", "clif_microbiology_nonculture_organism_category.csv"
    )
    if os.path.exists(nc_csv) and "microbiology_nonculture" in tables:
        df = pd.read_csv(nc_csv)
        tables["microbiology_nonculture"]["organism_category_to_group_mapping"] = dict(
            zip(df["organism_category"], df["organism_group"])
        )

    out_csv = os.path.join(a.mcide, "output", "clif_output_categories.csv")
    if os.path.exists(out_csv) and "output" in tables:
        df = pd.read_csv(out_csv)
        tables["output"]["output_category_to_group_mapping"] = dict(
            zip(df["output_category"], df["output_group"])
        )

    doc = {
        "clif_version": "3.0.0",
        "source": a.source_note
        or "CLIF 3.0 DDL/mCIDE (Common-Longitudinal-ICU-data-Format/CLIF, branch 3.0) "
        "via clifpy 3.0 schemas (Apache-2.0)",
        "tables": tables,
    }
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"wrote {len(tables)} tables to {a.out}")


if __name__ == "__main__":
    main()
