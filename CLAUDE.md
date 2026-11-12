# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository generates synthetic CLIF (Common Longitudinal ICU data Format) datasets for testing code without using protected health information (PHI). The synthetic data follows the CLIF 3.0.0 specification by default, and CLIF 2.1.0 with `--clif-version 2.1`, with realistic properties including missingness, outliers, and irregular measurement frequencies.

## Build and Run Commands

```bash
# Install dependencies
pip install -e .

# Run all tests
pytest

# Run specific test file
pytest tests/test_vitals.py

# Run single test
pytest tests/test_clif3.py -v

# Generate synthetic dataset (small test set: 10 patients, 12 hospitalizations)
python -m synthetic_clif --patients 10 --hospitalizations 12 --output data/test/

# Generate full dataset (10,000 hospitalizations)
python -m synthetic_clif --hospitalizations 10000 --output data/full/ -w 4

# Generate CLIF 2.1.0 tables (3.0.0 is the default)
python -m synthetic_clif --hospitalizations 100 --output data/v21/ --clif-version 2.1
```

## Architecture

### Core Modules

- **`synthetic_clif/config/`**: CLIF schema definitions and mCIDE permissible values loaded from CSVs
- **`synthetic_clif/generators/`**: Table-specific generators that produce pandas DataFrames
  - `base.py`: Abstract base generator with common logic for missingness, timestamps, outliers
  - `patient.py`: Demographics (patient table)
  - `hospitalization.py`: Hospital encounters with admission/discharge times
  - `vitals.py`: Vital signs with temporal autocorrelation
  - `labs.py`: Laboratory panels and when they are ordered; values come from the course
  - `medications.py`: Continuous and intermittent medication administration
  - `respiratory.py`: Respiratory support and ventilator settings
  - Additional generators for each CLIF table
- **`synthetic_clif/config/lab_catalog.py`**: one entry per lab category with its 2.1 and 3.0
  unit, order category, LOINC code and fallback range. Labs with no 2.1 unit are 3.0-only and
  are dropped from 2.1 output; add new labs here first
- **`synthetic_clif/clif3/`**: CLIF 3.0.0 support. `schema.py` loads the vendored
  `clif3_schema.json` (rebuild with `tools/build_clif3_schema.py`), `vocab.py`
  maps 2.1 values to 3.0, `convert.py` turns the generated 2.1 tables into 3.0
  tables (and derives airway, line, misc_icu_orders, ed_encounter), and
  `validate.py` checks any table against the 3.0 schema
- **`synthetic_clif/physiology/`**: `course.py` builds one hidden hourly `HospitalCourse` per
  hospitalization (severity, shock, respiratory failure, kidney injury, infection, devices,
  ventilator settings, vasopressor titration, vitals and lab values). The vitals, respiratory,
  continuous-medication and labs generators observe it via `course_for_row(hosp_row)`, adding
  only measurement noise. The course is seeded from a private `_course_seed` column that
  `SyntheticCLIFDataset` adds to a copy of the hospitalization table (with `_sex`); without it
  the seed is a hash of the hospitalization id, so standalone generators still agree.
  Keep any new per-stay state derivable from the hospitalization row, because parallel
  workers rebuild generators from (seed, mcide_dir) and only see their chunk.
  `events.py` adds slower clinical decisions to the course (infection and antibiotics,
  CRRT, proning, MCS, comfort care, lines, sedation/RASS/GCS, fluids, transfusions, code
  status), each from its own seeded stream so adding one never reshuffles the others.
  `outcomes.py` decides death and disposition; `SyntheticCLIFDataset._assign_outcomes` writes
  them before any other table is generated, and `course_for_row` rebuilds an expired stay
  with a terminal decline. Patient-level traits come from `_patient_seed`.
- **`synthetic_clif/models/`**: Dataclasses representing CLIF entities and patient state
- **`synthetic_clif/simulation/`**: Patient trajectory simulation engine
  - Maintains physiologically coherent state across time
  - Generates correlated observations (e.g., low BP triggers vasopressor administration)

### Data Flow

1. `PatientFactory` creates base patient demographics
2. `HospitalizationSimulator` generates admission/discharge with realistic LOS distribution
3. `TrajectoryEngine` simulates patient state over time, producing time-series events
4. Individual table generators transform trajectory events into CLIF-compliant DataFrames
5. `DatasetWriter` outputs parquet files per table

### Key Design Decisions

- All timestamps in UTC (YYYY-MM-DD HH:MM:SS+00:00)
- Physiological values use autoregressive models to prevent unrealistic jumps
- Missingness patterns vary by variable type (vitals more complete than assessments)
- Hospital length of stay follows log-normal distribution fitted to real ICU data
- Patient IDs use UUID format; hospitalization IDs include patient prefix for traceability

## CLIF 2.1.0 Schema Reference (internal simulation; 3.0.0 output comes from `synthetic_clif/clif3/`)

The CLIF format includes 16 beta tables (production-ready) and 12 concept tables (draft):

**Beta Tables**: patient, hospitalization, adt, vitals, labs, respiratory_support, medication_admin_continuous, medication_admin_intermittent, microbiology_culture, microbiology_susceptibility, patient_assessments, patient_procedures, hospital_diagnosis, crrt_therapy, position, code_status

**Concept Tables**: clinical_trial, ecmo_mcs, intake_output, invasive_hemodynamics, key_icu_orders, medication_orders, microbiology_nonculture, patient_diagnosis, place_based_index, provider, therapy_details, transfusion

## mCIDE Permissible Values

Standardized category values are defined in `synthetic_clif/config/mcide/`. Key categories:

- **vital_category**: temp_c, heart_rate, sbp, dbp, spo2, respiratory_rate, map, height_cm, weight_kg
- **device_category** (respiratory): IMV, NIPPV, CPAP, High Flow NC, Face Mask, Trach Collar, Nasal Cannula, Room Air, Other
- **discharge_category**: Home, SNF, Expired, Hospice, LTACH, Acute Care Hospital, AMA, etc.
- **location_category** (ADT): ed, ward, stepdown, icu, procedural, l&d, hospice, psych, rehab, radiology, dialysis, other

## Realistic Data Artifacts

The generator introduces these real-world artifacts:

1. **Missingness**: MCAR, MAR, and MNAR patterns based on clinical documentation practices
2. **Outliers**: Physiologically plausible extreme values (e.g., fever spikes) and documentation errors
3. **Irregular timing**: Vitals q1h in ICU, q4h on floor; labs ordered based on clinical need
4. **Temporal correlation**: Sequential readings follow autoregressive patterns
5. **Inter-variable correlation**: vitals, respiratory support, continuous meds and labs share one
   course per stay (see `synthetic_clif/physiology/`); `tests/test_physiology.py` checks it
