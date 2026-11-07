# synthetic_clif

Generate synthetic [CLIF (Common Longitudinal ICU Format)](https://clif-consortium.github.io/website/) 2.1.0 and 3.0.0 datasets for testing and development. Output is CLIF 2.1.0 by default; pass `--clif-version 3.0` for CLIF 3.0.0 (see [CLIF 3.0.0 output](#clif-300-output)).

## Quick Start: Download the Pre-generated Dataset

A pre-generated dataset with **10,000 hospitalizations** (~33 million rows across 28 tables) is available as a GitHub Release asset.

```bash
# Download and extract (requires GitHub CLI)
gh release download v0.6.0 -R AartikSarma/synthetic_clif -p "synth_clif_10k.tar.gz"
mkdir -p synth_clif_10k && tar -xzf synth_clif_10k.tar.gz -C synth_clif_10k

# Or download directly
curl -L https://github.com/AartikSarma/synthetic_clif/releases/download/v0.6.0/synth_clif_10k.tar.gz | tar -xz -C synth_clif_10k
```

Then load in Python:

```python
import pandas as pd

vitals = pd.read_parquet("synth_clif_10k/clif_vitals.parquet")
labs = pd.read_parquet("synth_clif_10k/clif_labs.parquet")
hospitalization = pd.read_parquet("synth_clif_10k/clif_hospitalization.parquet")
```

All files use the `clif_` prefix (e.g., `clif_vitals.parquet`, `clif_labs.parquet`).

## Purpose

This package creates realistic synthetic ICU data that follows the CLIF 2.1.0 specification. Because the data is entirely synthetic, it contains **no Protected Health Information (PHI)**, enabling:

- **Development on non-HIPAA compliant systems** - Use cloud IDEs, AI coding assistants (like Claude), and other tools that cannot access real patient data
- **Testing analysis pipelines** - Validate your CLIF analysis code before running on real data
- **Sharing reproducible examples** - Create datasets that can be freely shared for debugging and collaboration
- **CI/CD integration** - Run automated tests against synthetic data in any environment

## Installation

```bash
git clone https://github.com/AartikSarma/synthetic_clif.git
cd synthetic_clif
pip install -e .
```

## Generate Your Own Dataset

### Command Line

```bash
# Small test dataset
python -m synthetic_clif --patients 10 --hospitalizations 12 --output data/test/

# Full-size dataset (parallel generation with 4 workers)
python -m synthetic_clif --patients 8000 --hospitalizations 10000 --output data/full/ -w 4
```

### Command Line Options

```
Usage: python -m synthetic_clif [OPTIONS]

Options:
  --patients INT          Number of patients to generate (default: 10)
  --hospitalizations INT  Number of hospitalizations to generate (default: 12)
  --output PATH           Output directory for parquet files (default: data/)
  --seed INT              Random seed for reproducibility (default: 42)
  --format [parquet|csv]  Output format (default: parquet)
  --workers INT, -w INT   Number of parallel workers (default: 1)
  --no-concept-tables     Skip generating concept tables (draft status)
  --clif-version VERSION  CLIF version of the output: 2.1 (default) or 3.0
```

### Python API

```python
from synthetic_clif import SyntheticCLIFDataset
from pathlib import Path

dataset = SyntheticCLIFDataset(
    n_patients=100,
    n_hospitalizations=120,
    seed=42,
    include_concept_tables=True,
    workers=4,  # parallel generation (default: 1)
    clif_version="2.1",  # or "3.0" for CLIF 3.0.0 tables
)

tables = dataset.generate()

# Access individual tables as DataFrames
vitals_df = tables["vitals"]
labs_df = tables["labs"]

# Write to parquet files
dataset.to_parquet(Path("output/"))
```

## Generated Tables

The package generates all 28 CLIF 2.1.0 tables:

### Beta Tables (16)
| Table | Description |
|-------|-------------|
| patient | Patient demographics |
| hospitalization | Hospital encounters (admissions 2018-2024) |
| adt | Admission/discharge/transfer events |
| vitals | Vital signs (HR, BP, SpO2, temp, RR) |
| labs | Laboratory results (52 categories) |
| respiratory_support | Ventilator settings and oxygen delivery |
| medication_admin_continuous | IV infusions (vasopressors, sedatives) |
| medication_admin_intermittent | Scheduled and PRN medications |
| microbiology_culture | Culture results and organisms |
| microbiology_susceptibility | Antibiotic susceptibilities |
| patient_assessments | GCS, RASS, CAM-ICU, pain scores |
| patient_procedures | ICD-10-PCS and CPT procedures |
| hospital_diagnosis | ICD-10-CM diagnoses |
| code_status | DNR/DNI and comfort care orders |
| position | Patient positioning (prone/supine) |
| crrt_therapy | Continuous renal replacement therapy |

### Concept Tables (12)
| Table | Description |
|-------|-------------|
| clinical_trial | Trial enrollment |
| ecmo_mcs | ECMO and mechanical circulatory support |
| intake_output | Fluid balance |
| invasive_hemodynamics | PA catheter and arterial line data |
| key_icu_orders | PT/OT/Speech evaluations |
| medication_orders | Prescription orders |
| microbiology_nonculture | PCR and rapid diagnostics |
| patient_diagnosis | Problem list diagnoses |
| place_based_index | ADI and SVI geographic indices |
| provider | Care team assignments |
| therapy_details | PT/OT/Speech session details |
| transfusion | Blood product administration |

## CLIF 3.0.0 Output

`--clif-version 3.0` (or `clif_version="3.0"`) writes CLIF 3.0.0 tables. The
generators still simulate in 2.1 terms; `synthetic_clif/clif3/` then converts
the tables to 3.0.0 with the same seed, so patient and hospitalization IDs match
a 2.1 run.

- **Vocabulary:** every `*_category`, `*_group` and `*_type` value is lowercase
  snake_case, with the official 2.1 to 3.0 renames (for example `High Flow NC`
  to `hfnc`, `Long Term Care Hospital (LTACH)` to `ltach`).
- **Renamed tables:** `ecmo_mcs` becomes `mcs` (long format, one row per
  setting); `intake_output` splits into `input` and `output`; `key_icu_orders`
  becomes `consult_orders`.
- **New columns:** `adt.room_id`/`bed_id`, `vitals.measured_dttm`,
  `labs.loinc_version`, `administering_provider_id` on medication
  administrations (linked to the `provider` table), `patient.language_interpreter`
  and others. Troponin is reported in ng/L.
- **New tables derived from existing data:** `airway` (from IMV episodes and
  tracheostomy), `misc_icu_orders` (extubation orders), `line` (from
  vasopressors, CRRT, invasive hemodynamics) and `ed_encounter` (from ED ADT rows).
- **Not yet generated:** `drain`, `intermittent_dialysis`, `patient_attributes`,
  `radiology`, `clinical_notes_facts`, `validated_diagnosis`, `model_registry`
  and `scores`. PVR and SvO2 rows of `invasive_hemodynamics` are dropped because
  3.0 has no category for them.

The 3.0.0 schema (columns, types, permissible values, group mappings) is
vendored in `synthetic_clif/clif3/clif3_schema.json`, built by
`tools/build_clif3_schema.py` from the CLIF `3.0` branch as packaged in
clifpy. A built-in check covers all 41 tables, including the ones clifpy has no
table class for yet:

```python
from synthetic_clif.clif3.validate import validate_tables
errors = validate_tables(dataset.generate())  # {table: []} when valid
```

## Data Characteristics

The synthetic data includes realistic artifacts found in real EHR data:

- **Temporal autocorrelation** - Sequential vital signs follow physiologically plausible patterns
- **Inter-variable correlation** - Low SpO2 triggers respiratory support escalation; low MAP triggers vasopressors
- **Irregular measurement frequency** - ICU vitals ~hourly, ward vitals ~q4h
- **Missingness patterns** - MCAR, MAR, and MNAR based on clinical documentation practices
- **Outliers** - Physiologically plausible extremes (fever spikes, hypotensive episodes)
- **Variable length of stay** - Log-normal distribution (median ~6.4 days, range 1-60+)
- **Consortium-calibrated distributions** - Demographics, ventilator settings, and clinical parameters match the 13-institution CLIF consortium aggregate
- **Specific ICU locations** - ADT uses MICU, SICU, CCU, NICU (not generic "ICU")
- **Validated against CLIFPy** - Schema, data types, categorical values, and units verified

## Running Tests

```bash
# Run all tests (includes CLIFPy validation)
pytest

# Run only fast unit tests
pytest tests/ --ignore=tests/test_clifpy_validation.py

# Run only CLIFPy validation
pytest tests/test_clifpy_validation.py -v
```

Tests require `clifpy` for schema validation: `pip install clifpy`

## CI/CD

- **CI** runs on every push/PR: pytest + CLIFPy validation
- **Release workflow** generates and uploads the 10K dataset when a new release is published

To create a new release with an updated dataset:

```bash
gh release create v0.x.0 --title "v0.x.0 - description" --notes "Release notes"
```

The release-dataset workflow will automatically generate the 10K dataset and attach it as `synth_clif_10k.tar.gz`.

## License

MIT

## Related Projects

- [CLIF Consortium](https://clif-consortium.github.io/website/) - The Common Longitudinal ICU Format specification
- [CLIF GitHub](https://github.com/clif-consortium) - Official CLIF tools and documentation
- [CLIFPy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) - Python validation package for CLIF data
