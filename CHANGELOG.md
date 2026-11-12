# Changelog

## Unreleased: validation tests that fail on bad data (#53)

Before this change the clifpy tests filtered errors on type names clifpy no
longer uses, so they matched nothing and passed whatever the data looked
like. Nothing checked keys, timestamps, impossible values or units.

Now the tests fail when the output breaks the CLIF rules, and fixing what
they found changed some of the generated data.

### New checks

`synthetic_clif.validation.check_dataset(tables, clif_version)` returns a
list of issues (empty means the dataset passed). It runs on 2.1 and 3.0
output and uses the rules of the consortium validator clifpy, vendored in
`synthetic_clif/validation/clif_rules.json` (rebuild with
`tools/build_clif_rules.py`):

- `schema` (3.0): columns, dtypes, vocabularies, category groups.
- `primary_key`: duplicate rows on a table's key.
- `foreign_key`: ids that point at nothing, and patient_id that disagrees
  with the hospitalization.
- `chronology`: an end before its start (admission and discharge, ADT in
  and out, lab order, collect and result, any `*start*`/`*end*` pair).
- `overlap`: overlapping ADT stays or hospitalizations of one patient.
- `stay_window`: a timestamp outside its stay, or after death.
- `value_range`: a number outside clifpy's outlier range for its category.
- `units`: a lab unit that is not the CLIF unit, or a continuous medication
  dose that is not a rate.
- `death`: an expired discharge with no matching death time.
- `never_charted`: a required setting (say, high flow liter flow) missing
  from every row of that device.

`tests/test_integrity.py` runs every check on both versions, injects one
error of each kind to show the check catches it, and asserts the range rules
cover the generated values. The clifpy tests now use a fixed error filter
(`tests/clifpy_errors.py`) and a test that injects errors into it, so a
future clifpy rename fails loudly. The 3.0 validator now checks every dtype,
and 3.0 conversion raises instead of turning uncastable values into nulls.

New tables are covered by the generic checks (ids, `*_dttm` columns,
start/end pairs) with no change; table-specific rules go in the `EXTRA_*`
settings in `synthetic_clif/validation/integrity.py`.

### Changes users will notice

- 2.1 `ecmo_mcs` columns now match the CLIF 2.1 schema: `device_metric_name`,
  `device_rate`, `sweep`, `flow`, `fdO2` (was `control_parameter_name`,
  `control_parameter_value`, `sweep_set`, `fdO2_set`).
- 2.1 troponin I is reported in ng/L, the CLIF unit (values x1000).
- High flow nasal cannula flow is in `lpm_set`, not `flow_rate_set`.
- CPAP and NIPPV rows now record observed respiratory rate and tidal volume.

### Fixes found by the new checks

- Overlapping ADT stays when an ICU stay followed a stepdown stay.
- Readmissions that started before the previous discharge.
- Labs collected after discharge or death; 3.0 labs ordered before
  admission; duplicate 2.1 lab rows on the same key.
- Microbiology collected after discharge.
- Medication orders written before admission.
- CRRT blood flow below 150 mL/min.

### Not yet covered

- Medication dose ranges: clifpy has ranges for about half the doses
  generated; the rest are checked for unit kind only.
- 3.0 tables not yet generated get only the generic checks once they are.

## Unreleased: one shared hospital course (PR #60)

Before this change each table drew its own values. MAP was generated
independently of SBP and DBP, pressors ran whatever the blood pressure,
ventilator settings were re-drawn every hour, labs had no memory from one
day to the next, and tables such as transfusion, ECMO, consults and the
problem list were random rows unrelated to the rest of the stay.

Now every hospitalization has one hidden hourly course
(`synthetic_clif/physiology/course.py`). Clinical decisions are layered on
it (`physiology/events.py`) and outcomes follow from it
(`physiology/outcomes.py`). Each generator observes the course and adds
only measurement noise, so the tables agree with each other.

### Relationships added

**Vitals and monitoring**
- MAP = DBP + pulse pressure / 3, from the same reading as SBP and DBP.
  A dropped or distorted reading affects all three together.
- Heart rate, temperature, respiratory rate and SpO2 follow the course's
  shock, infection and respiratory severity.
- An arterial line appears when the patient is unstable.
- Height is a patient trait, constant across readmissions; weight follows
  a per-patient BMI.

**Respiratory support**
- Devices escalate and wean with respiratory severity (room air, nasal
  cannula, high flow, NIPPV, IMV).
- FiO2 is titrated to oxygenation; SpO2 and PaO2 follow it.
- Ventilator settings persist through an episode: tidal volume 6-8 mL/kg
  predicted body weight, PEEP from the ARDSNet table.
- Plateau = PEEP + VT / compliance, peak = plateau + resistive pressure,
  minute ventilation = VT x RR, observed RR is at least set RR.
- Prone positioning for severe hypoxemia on IMV.

**Medications**
- Vasopressors titrate to a MAP goal and start, change and stop with shock.
- Sedation and analgesia run per IMV episode; RASS, GCS and CAM-ICU follow
  sedation depth and delirium (CAM is "unable to assess" when deeply sedated).
- Infection leads to cultures, then antibiotics, with renal dose adjustment.
- No administration is recorded after discharge.
- `medication_orders` is built from the administration tables: one order
  per `med_order_id`, with the first or modal dose, the frequency seen
  between doses, PRN status for as-needed drugs, and an end time and status
  that follow the last dose and the discharge.

**Labs**
- Per-patient baselines with day-to-day autocorrelation.
- BUN tracks creatinine; chloride follows the anion gap; pH comes from
  Henderson-Hasselbalch.
- Serial lactates on pressors and blood gases on IMV.
- Patients with chronic kidney disease have a higher baseline creatinine.

**Organ support and procedures**
- CRRT for severe kidney injury, with matching intake/output.
- ECMO, Impella, IABP and CentriMag for refractory shock or hypoxemia, with
  settings that wean (flow, sweep, FdO2, P-level, augmentation ratio).
- Pulmonary artery catheters during severe shock; invasive hemodynamics are
  recorded only while a catheter is in.
- Central lines and other procedures timed to the course.

**Blood products**
- Transfusions follow low hemoglobin, platelets or high INR (and bleeding).
- One row per unit, with product codes and irradiated or leukoreduced
  attributes; units of one order hang back to back.
- The same units appear as blood product volume in `intake_output`.

**Intake and output**
- Fluid boluses, maintenance fluids and tube feeds come from the course.
  The feed formula follows the patient (renal formula for CKD or CRRT,
  diabetic formula for diabetes).
- Urine output is hourly for critically ill patients and per shift otherwise.

**Location, outcomes and code status**
- ADT ICU windows cover all IMV, vasopressor and CRRT time.
- Death, discharge disposition and death after discharge depend on peak
  illness, organ support and age. Only a patient's last stay can end in
  death, and the death date matches it.
- Code status changes (DNR, comfort care) precede many deaths.

**Concept tables**
- `key_icu_orders`: PT/OT once the patient is stable and awake; speech and
  swallow evaluation after more than 48 h of IMV; nutrition when tube
  feeds start; palliative care for deaths, comfort care, older DNR patients
  and hospice discharges; social work for discharges other than home.
- `therapy_details`: an evaluation after each therapy order, then treatment
  sessions on stable days; respiratory therapy while on high flow, NIPPV or
  IMV.
- `microbiology_nonculture`: respiratory viral PCR at presentation, mostly
  for respiratory illness, positive mainly in lung infections; C. difficile
  testing after several days of antibiotics.
- `patient_diagnosis`: diabetes, COPD, CKD and obesity come from the same
  patient traits that shape labs and vitals; hypertension, coronary
  disease, heart failure and atrial fibrillation cluster with them and with
  age. Onset dates fall before the first admission.

### Changes users will notice

- Outcome rates now come from illness severity. In a 400-stay sample:
  mortality about 12%, IMV 35%, vasopressors 31%, CRRT 4.5%, proning 5.5%,
  mechanical circulatory support 1.8%, RBC transfusion 12-15%.
- Row counts and shapes of several concept tables changed (transfusion is
  one row per unit; ECMO and invasive hemodynamics have rows only during
  support; intake/output and therapy sessions are denser for sicker patients).
- Birth dates are anchored to the end of the CLIF date range instead of
  today's date, so the same seed gives the same dataset on any day.
- Influenza PCR results are produced only for CLIF 3.0 output; CLIF 2.1's
  non-culture vocabulary has no influenza.
- `EcmoMcsGenerator.generate(ecmo_rate=...)` and
  `InvasiveHemodynamicsGenerator.generate(pa_catheter_rate=...)` still
  accept these arguments but ignore them; the course decides who needs
  support.
- New optional arguments: `TherapyDetailsGenerator.generate(orders_df=...)`
  and `MicrobiologyNoncultureGenerator.generate(include_influenza=...)`.
- Standalone generators without the dataset's private `_course_seed` fall
  back to a hash of the hospitalization id, so they remain reproducible.

### Fixes

- Medication administrations no longer land after discharge.
- Birth dates no longer depend on the date the generator runs.
- Parallel generation starts workers with forkserver (or spawn), which
  fixes a hang when forking from the threaded vitals, labs and respiratory
  generators.

### Not yet covered

- `provider`, `place_based_index` and `clinical_trial` are still random and
  do not follow the course.
- The CLIF 3.0 line table is derived from medications and IMV, not from the
  course's line placements.
- Hematocrit is not generated.
- CRRT net ultrafiltration is not written to the output table.
- Influenza results are not available in CLIF 2.1 output.

### Checks

160 tests pass, including 28 new coherence tests in
`tests/test_physiology.py`. CLIF 3.0 output passes the schema validator.
