# Changelog

## Unreleased: the eight new CLIF 3.0 tables

Before this change, CLIF 3.0 output left out eight tables that 3.0 adds:
`intermittent_dialysis`, `drain`, `patient_attributes`, `radiology`,
`clinical_notes_facts`, `validated_diagnosis`, `model_registry` and
`scores`.

Now all 41 tables of the 3.0 schema are generated. The eight new tables are
built natively in 3.0 terms (`synthetic_clif/generators/v3/`) from the same
hospital course as every other table, so they agree with the rest of the
stay. They are only produced for 3.0 output; 2.1 output is unchanged.

### Relationships added

**intermittent_dialysis**
- A new course event: kidney injury that does not need CRRT can be treated
  with intermittent haemodialysis once the patient has been off pressors
  for six hours. Sessions run 3-4 hours about every other day until the
  kidneys recover, and each one lowers creatinine and potassium.
- Every haemodialysis stay has a dialysis catheter in `line` and a billed
  haemodialysis procedure (5A1D70Z / 90935). Ultrafiltration is charted in
  `output` as `ultrafiltration`.

**drain**
- Foley catheters cover ICU time (and some ward stays), followed at times
  by an external catheter; an orogastric or nasogastric tube with every
  intubation; post-pyloric feeding tubes when feeds start; a PEG for about
  half of tracheostomy patients; Jackson-Pratt drains for abdominal
  infections; chest tubes and pigtails wherever `patient_procedures` bills
  a thoracostomy.
- Devices still in place at death, and PEG tubes, have no removal time.
- Urine output in `output` is labelled by the device in place
  (`indwelling_urinary_catheter`, `external_urinary_catheter` or
  `urethral`) and carries that drain's id in `lda_id`. Chest tube output is
  charted too.

**patient_attributes**
- Smoking status follows COPD; employment and marital status follow age;
  functional status follows frailty; social-needs answers cluster.
- Registration and nursing history are charted at each stay, pregnancy
  status for women aged 12-50, and social-needs screening grows more common
  in later years. Smokers may quit between stays, and function worsens
  after a stay that ends in a nursing facility, LTACH or rehab.

**radiology**
- A chest film on arrival, after each intubation and line placement, after
  chest tubes go in and come out, and most mornings on the ventilator; a
  film to check feeding tube position; CT to find an infection's source
  (without contrast when the kidneys are injured); CT head for coma or
  delirium; ultrasound for shock liver and kidney injury; echocardiograms
  in shock; Dopplers in long stays.
- Studies are portable in the ICU. Order, capture and report times follow
  in that order. Each report is a short templated impression that matches
  the course (tube positions, bilateral opacities in severe respiratory
  failure, low ejection fraction in cardiogenic shock) and is marked
  `[synthetic report]`.

**clinical_notes_facts**
- History and physical, daily progress notes (ICU team or hospital
  medicine, written by a trainee and co-signed), consult notes for each
  consult order, a nephrology note when dialysis starts, procedure notes
  for each procedure by its performing provider, transfer notes, nursing
  and pharmacy notes, case-management notes for non-home discharges, and a
  discharge summary for survivors or a death note for those who died.
- Each note has revisions (incomplete, signed, co-signed, addendum) in time
  order. Physician authors come from the `provider` table.

**validated_diagnosis**
- About half of ICU-level stays are chart-reviewed for sepsis (Sepsis-3),
  acute respiratory failure, ARDS (Berlin) and acute kidney injury
  (KDIGO), from the same course that drives the billed diagnoses. Every
  confirmed sepsis label has a billed A41 code.
- Labels the course does not support are `ruled_out`; ARDS in cardiogenic
  shock and a few others are `uncertain` and go to a consensus panel.
  Reviews are dated one to twelve months after discharge.

**model_registry and scores**
- The registry lists two versions each of the Deterioration Index and the
  Sepsis Model, and a fall-risk model, with release dates. Each score comes
  from the version live on its date.
- The Deterioration Index (hourly) rises with severity, oxygen support and
  pressors, so it is higher in patients who die. The Sepsis Model (every
  four hours) follows infection; the older version is noisier. Fall risk
  (each shift) rises with age and delirium and falls on the ventilator.

### Other changes

- The ARDS label in `validated_diagnosis` needs PEEP of at least 5 and
  bilateral opacities, so it agrees with the chest film reports.
- `SyntheticCLIFDataset` keeps the natively generated 3.0 tables in
  `_tables_v3_native`, and `convert_to_clif3(..., native_tables=...)`
  accepts them.
- `HospitalCourse` gains `ihd_sessions`, `peg_h` and `abd_drain`.

### Not yet covered

- External ventricular drains are not generated.
- `attribute_group` follows the mCIDE mapping, but the vendored schema
  does not yet check it.

### Checks

35 new tests in `tests/test_clif3_new_tables.py` check each new table
against the 3.0 schema and against the tables it is tied to. All 41 tables
pass the built-in 3.0 validator.

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
