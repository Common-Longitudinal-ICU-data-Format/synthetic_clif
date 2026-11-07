"""Clinical events layered on a HospitalCourse.

The base course (``course.py``) builds the patient's physiology and the
bedside support that responds to it minute by minute: oxygen, ventilator,
vasopressors, sedation. This module adds the decisions clinicians make on a
slower clock, each driven by that physiology:

- infection recognition, cultures and antibiotics
- bleeding and transfusion (RBC, platelets, plasma, cryoprecipitate)
- CRRT for severe kidney injury, which clears creatinine and potassium
- prone positioning for severe hypoxaemia, which raises PaO2
- ECMO and other mechanical circulatory support
- comfort care before death, with vasopressors withdrawn
- arterial, central, dialysis and pulmonary-artery lines
- sedation depth, RASS, GCS and delirium
- fluid boluses, maintenance fluid, tube feeds, meals and urine output
- code status

Every event uses its own random stream derived from the course seed, so
adding or changing one event never reshuffles the others or the base course.
"""

from __future__ import annotations

import math

import numpy as np

from synthetic_clif.physiology.course import _lag, _ou, _stable_hash, severinghaus_spo2


def _stream(course, name: str) -> np.random.Generator:
    return np.random.default_rng((course.rng_seed + _stable_hash("events:" + name)) & 0xFFFFFFFF)


def _runs(mask) -> list[tuple[int, int]]:
    """[(start, end)) index ranges where mask is true."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _sustained(mask, k: int) -> np.ndarray:
    """True where mask has been true for the last k hours."""
    out = np.zeros(len(mask), dtype=bool)
    streak = 0
    for i, v in enumerate(mask):
        streak = streak + 1 if v else 0
        out[i] = streak >= k
    return out


def build_events(c) -> None:
    n = c.n_hours
    _bleeding(c, _stream(c, "bleed"))
    _crrt(c, _stream(c, "crrt"))
    _prone(c, _stream(c, "prone"))
    _mcs(c, _stream(c, "mcs"))
    _comfort(c, _stream(c, "comfort"))
    _infection_and_antibiotics(c, _stream(c, "infection"))
    _shock_type(c)
    _lines(c, _stream(c, "lines"))
    _sedation(c, _stream(c, "sedation"))
    _fluids(c, _stream(c, "fluids"))
    _transfusions(c, _stream(c, "transfusion"))
    _code_status(c, _stream(c, "code"))
    assert len(c.rass) == n


# ----------------------------------------------------------------------
# Infection, cultures, antibiotics
# ----------------------------------------------------------------------
SOURCES = ["pulmonary", "urinary", "bloodstream", "abdominal", "skin"]
SOURCE_P = [0.45, 0.20, 0.12, 0.13, 0.10]
SOURCE_FLUID = {"pulmonary": "respiratory", "urinary": "urine", "bloodstream": "blood",
                "abdominal": "blood", "skin": "wound"}


def _infection_and_antibiotics(c, rng):
    n = c.n_hours
    suspected_h = None
    if c.infected:
        c.infection_source = str(rng.choice(SOURCES, p=SOURCE_P))
        gram_neg_p = {"pulmonary": 0.55, "urinary": 0.8, "bloodstream": 0.45,
                      "abdominal": 0.8, "skin": 0.2}[c.infection_source]
        c.pathogen_gram = "gram_neg" if rng.random() < gram_neg_p else "gram_pos"
        above = np.nonzero(c.infection > 0.2)[0]
        if len(above):
            suspected_h = float(above[0]) + rng.uniform(0, 3)
    else:
        # Sick but not infected: often treated empirically, then stopped
        sick = np.nonzero(c.severity > 0.45)[0]
        if len(sick) and rng.random() < 0.45:
            suspected_h = float(sick[0]) + rng.uniform(0, 4)
    if suspected_h is None or suspected_h >= n - 1:
        return
    c.infection_recognized_h = suspected_h
    true_inf = c.infected

    def culture_set(h, reason):
        on_vent = c.level[min(int(h), n - 1)] == 4
        bacteremic = true_inf and (c.infection_source == "bloodstream" or rng.random() < 0.3)
        sets = []
        for k in range(2):  # two blood culture sets
            pos = (bacteremic and reason == "initial") or (rng.random() < 0.03)
            sets.append(dict(h=h + 0.05 * k, fluid="blood", positive=bool(pos),
                             contaminant=bool(pos and not bacteremic)))
        if rng.random() < 0.75 or (true_inf and c.infection_source == "urinary"):
            pos = (true_inf and c.infection_source == "urinary" and reason == "initial"
                   and rng.random() < 0.85) or rng.random() < 0.06
            sets.append(dict(h=h + rng.uniform(0, 2), fluid="urine", positive=bool(pos), contaminant=False))
        if on_vent or (true_inf and c.infection_source == "pulmonary" and rng.random() < 0.5):
            pos = (true_inf and c.infection_source == "pulmonary" and reason == "initial"
                   and rng.random() < 0.7) or (on_vent and rng.random() < 0.1)
            sets.append(dict(h=h + rng.uniform(0, 3), fluid="respiratory", positive=bool(pos), contaminant=False))
        if true_inf and c.infection_source == "skin" and reason == "initial":
            sets.append(dict(h=h + rng.uniform(0, 6), fluid="wound", positive=rng.random() < 0.8,
                             contaminant=False))
        return sets

    cultures = culture_set(suspected_h, "initial")
    # Fever later in the stay prompts repeat blood cultures
    last = suspected_h
    for i in range(int(suspected_h) + 48, n - 2):
        if c.temp[i] > 38.3 and i - last >= 48:
            cultures += culture_set(float(i) + rng.uniform(0, 1), "fever")
            last = i
    c.cultures = [x for x in cultures if x["h"] < n - 1]

    # Empiric antibiotics within hours of recognition, faster in shock
    shocked = c.shock[int(suspected_h)] > 0.5
    start = suspected_h + rng.uniform(0.3, 1.5 if shocked else 4.0)
    end_cap = float(n - 1)
    if c.comfort_h is not None:
        end_cap = min(end_cap, c.comfort_h)
    if true_inf:
        course_len = rng.uniform(5, 10) * 24
    else:
        course_len = rng.uniform(36, 72)
    beta = str(rng.choice(["piperacillin_tazobactam", "cefepime", "meropenem"],
                          p=[0.5, 0.3, 0.2] if not shocked else [0.4, 0.3, 0.3]))
    abx = [dict(agent=beta, start=start, end=min(start + course_len, end_cap))]
    if rng.random() < 0.8:
        vanc_len = course_len
        # Stop vancomycin once cultures show no gram-positive pathogen
        if not (true_inf and c.pathogen_gram == "gram_pos"):
            vanc_len = min(course_len, rng.uniform(48, 72))
        abx.append(dict(agent="vancomycin", start=start + rng.uniform(0, 1), end=min(start + vanc_len, end_cap)))
    c.antibiotics = [a for a in abx if a["end"] > a["start"]]


# ----------------------------------------------------------------------
# Bleeding, transfusion
# ----------------------------------------------------------------------
def _bleeding(c, rng):
    n = c.n_hours
    bleed = np.zeros(n)
    if rng.random() < 0.05 + 0.04 * (c.acuity >= 3):
        h0 = int(rng.uniform(0, max(n - 12, 1)))
        drop = rng.uniform(2.0, 4.5)
        ramp = np.clip((np.arange(n) - h0) / rng.uniform(4, 12), 0, 1)
        bleed = drop * ramp
    c.bleed = bleed


def _transfusions(c, rng):
    n = c.n_hours
    c.transfusions = []
    last_end = c.comfort_h if c.comfort_h is not None else n - 1
    checks = np.arange(rng.uniform(2, 8), last_end, 12.0)
    for h in checks:
        hgb = c._hgb_raw(h) + c.product_effect(h + 1, "Packed RBCs", 1.0, None)
        if hgb < 7.0:
            units = 2 if hgb < 6.0 else 1
            c.transfusions.append(dict(h=float(h + rng.uniform(0.5, 2)), component="Packed RBCs", units=units))
            if units >= 2 and c.bleed[int(h)] > 1.0 and rng.random() < 0.6:
                c.transfusions.append(dict(h=float(h + rng.uniform(1, 3)), component="Fresh Frozen Plasma", units=2))
                if rng.random() < 0.3:
                    c.transfusions.append(dict(h=float(h + rng.uniform(1, 4)), component="Cryoprecipitate", units=1))
        plt = c._plt_raw(h) + c.product_effect(h + 1, "Platelets", 30.0, 72.0)
        if plt < 20 or (plt < 50 and c.bleed[int(h)] > 1.0):
            c.transfusions.append(dict(h=float(h + rng.uniform(0.5, 3)), component="Platelets", units=1))
        inr = c._inr_raw(h) - c.product_effect(h + 1, "Fresh Frozen Plasma", 0.25, 12.0)
        if inr > 2.5 and rng.random() < 0.3:
            c.transfusions.append(dict(h=float(h + rng.uniform(0.5, 3)), component="Fresh Frozen Plasma", units=2))
    c.transfusions.sort(key=lambda x: x["h"])


# ----------------------------------------------------------------------
# Kidney replacement
# ----------------------------------------------------------------------
def _crrt(c, rng):
    n = c.n_hours
    crrt = np.zeros(n, dtype=bool)
    c.crrt_mode = str(rng.choice(["cvvhdf", "cvvhd", "cvvh"], p=[0.5, 0.35, 0.15]))
    need = _sustained(c.aki > 0.6, 12)
    candidate = rng.random() < 0.75
    if candidate and need.any():
        s = int(np.argmax(need) + rng.uniform(2, 10))
        if s < n - 2:
            recovered = _sustained(c.aki < 0.45, 24)
            after = np.nonzero(recovered[s:])[0]
            e = s + int(after[0]) if len(after) else n
            crrt[s:e] = True
    c.crrt = crrt
    c._crrt_eff = np.clip(_lag(crrt.astype(float), 8.0), 0, 1)


# ----------------------------------------------------------------------
# Prone positioning
# ----------------------------------------------------------------------
def _prone(c, rng):
    n = c.n_hours
    prone = np.zeros(n, dtype=bool)
    on_imv = c.level == 4
    severe = on_imv & (c.pf_true < 150) & (c.fio2 >= 0.6)
    eligible = _sustained(severe, 12)
    if eligible.any() and rng.random() < 0.45:
        i = int(np.argmax(eligible) + rng.uniform(1, 6))
        while i < n:
            if not on_imv[min(i, n - 1)]:
                break
            if c.pf_true[i] >= 150 and c.fio2[i] <= 0.6:
                break
            e = min(i + int(rng.uniform(15, 18)), n)
            if not on_imv[i:e].all():
                e = i + int(np.argmin(on_imv[i:e]))
            prone[i:e] = True
            i = e + int(rng.uniform(6, 9))
    c.prone = prone
    if prone.any():
        boost = np.where(prone, 1.35, 1.0)
        c.pao2 = np.clip(c.pao2 * boost, 35, 550)
        c.spo2 = np.clip(severinghaus_spo2(c.pao2), 60, 100)


# ----------------------------------------------------------------------
# ECMO and mechanical circulatory support
# ----------------------------------------------------------------------
def _mcs(c, rng):
    n = c.n_hours
    c.mcs = []
    on_imv = c.level == 4
    if c.age < 65:
        need = _sustained(on_imv & (c.pf_true < 80), 6)
        if need.any() and rng.random() < 0.5:
            s = int(np.argmax(need) + rng.uniform(2, 8))
            if s < n - 6:
                better = _sustained(c.pf_true > 150, 24)
                after = np.nonzero(better[s:])[0]
                e = min(s + int(after[0]) if len(after) else n, n)
                c.mcs.append(dict(device="VV_ECMO", start=s, end=e))
                _lung_rest(c, rng, s, e)
    if c.age < 70 and not c.mcs:
        need = _sustained((c.shock > 1.1) & (c.ne_eq > 0.4), 3)
        if need.any() and rng.random() < 0.25:
            s = int(np.argmax(need) + rng.uniform(1, 4))
            if s < n - 6:
                better = _sustained(c.shock < 0.6, 24)
                after = np.nonzero(better[s:])[0]
                e = min(s + int(after[0]) if len(after) else n, n)
                c.mcs.append(dict(device="VA_ECMO", start=s, end=e))
                c.map[s:e] = np.maximum(c.map[s:e], 65 + _ou(rng, e - s, 3, 0.8))
                _set_bp_from_map(c, s, e)
    if c.dobutamine.any() and not c.mcs and rng.random() < 0.5:
        s = int(np.argmax(c.dobutamine) + rng.uniform(1, 12))
        if s < n - 6:
            device = str(rng.choice(["Impella_CP", "IABP", "Impella_5.5"], p=[0.5, 0.35, 0.15]))
            e = min(s + int(rng.uniform(48, 144)), n)
            c.mcs.append(dict(device=device, start=s, end=e))


def _lung_rest(c, rng, s, e):
    """Ultra-protective ventilation while VV ECMO does the gas exchange."""
    vt = max(round(4 * c.pbw_kg / 10) * 10, 250)
    for i in range(s, e):
        if c.level[i] != 4:
            continue
        c.fio2[i] = 0.4
        c.peep[i] = 10.0
        if not np.isnan(c.rr_set[i]):
            c.rr_set[i] = 10.0
        if not np.isnan(c.vt_set[i]):
            c.vt_set[i] = vt
    span = slice(s, e)
    c.pao2[span] = np.clip(rng.uniform(65, 90) + _ou(rng, e - s, 8, 0.7), 50, 200)
    c.spo2[span] = np.clip(severinghaus_spo2(c.pao2[span]), 60, 100)
    c.rr[span] = np.where(np.isnan(c.rr_set[span]), c.rr[span],
                          np.maximum(c.rr[span], c.rr_set[span]))


def _set_bp_from_map(c, s, e):
    pp = c.sbp[s:e] - c.dbp[s:e]
    c.dbp[s:e] = c.map[s:e] - pp / 3
    c.sbp[s:e] = c.map[s:e] + 2 * pp / 3


# ----------------------------------------------------------------------
# Comfort care before death
# ----------------------------------------------------------------------
def _comfort(c, rng):
    n = c.n_hours
    c.comfort_h = None
    if not c.expired or rng.random() > 0.6:
        return
    w = max(int(n - 2 - rng.uniform(1, 8)), 0)
    c.comfort_h = w
    c.ne_eq[w:] = 0.0
    c.vasopressin[w:] = False
    c.dobutamine[w:] = False
    c.map[w:] = c._map_u[w:]
    _set_bp_from_map(c, w, n)
    c.crrt[w:] = False
    c._crrt_eff = np.clip(_lag(c.crrt.astype(float), 8.0), 0, 1)
    c.insulin[w:] = False


def _shock_type(c):
    if c.dobutamine.any() or any(m["device"] in ("Impella_CP", "IABP", "Impella_5.5", "VA_ECMO") for m in c.mcs):
        c.shock_type = "cardiogenic"
    elif c.infected:
        c.shock_type = "distributive"
    elif c.bleed is not None and c.bleed.max() > 1.5:
        c.shock_type = "hypovolemic"
    else:
        c.shock_type = "distributive"


# ----------------------------------------------------------------------
# Lines
# ----------------------------------------------------------------------
def _merge(runs, gap):
    out = []
    for s, e in runs:
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _lines(c, rng):
    n = c.n_hours
    art = np.zeros(n, dtype=bool)
    cvc = np.zeros(n, dtype=bool)
    dial = np.zeros(n, dtype=bool)
    pac = np.zeros(n, dtype=bool)
    unstable = (c.ne_eq > 0) | (c.level == 4) | c.crrt
    for k, (s, e) in enumerate(_merge(_runs(unstable), 12)):
        if rng.random() < (0.85 if k == 0 else 0.7):
            a = int(s + rng.uniform(0, 1.5))
            b = int(min(e + rng.uniform(6, 30), n))
            art[a:b] = True
    if not unstable.any() and c.acuity >= 3 and rng.random() < 0.15:
        a = int(rng.uniform(0, 6))
        art[a:int(min(a + rng.uniform(24, 72), n))] = True
    for s, e in _merge(_runs(c.ne_eq > 0), 24):
        if rng.random() < 0.9:
            a = int(s + rng.uniform(0.5, 6))
            b = n if rng.random() < 0.3 else int(min(e + rng.uniform(24, 96), n))
            cvc[a:b] = True
    for s, e in _runs(c.crrt):
        a = max(int(s - rng.uniform(1, 3)), 0)
        dial[a:int(min(e + rng.uniform(6, 48), n))] = True
    cardiogenic = c.dobutamine.any()
    severe = c.ne_eq > 0.3
    if (cardiogenic and rng.random() < 0.6) or (severe.any() and rng.random() < 0.15):
        first = int(np.argmax(c.dobutamine if cardiogenic else severe))
        a = int(first + rng.uniform(2, 12))
        pac[a:int(min(a + rng.uniform(48, 96), n))] = True
    c.art_line, c.central_line, c.dialysis_line, c.pa_catheter = art, cvc, dial, pac


# ----------------------------------------------------------------------
# Sedation, RASS, GCS, delirium
# ----------------------------------------------------------------------
def _sedation(c, rng):
    n = c.n_hours
    on_imv = c.level == 4
    sedated = on_imv & (c.trach == 0)
    since_intub = np.zeros(n)
    for s, e in c.imv_episodes:
        since_intub[s:e] = np.arange(e - s)
    paralysed_like = c.prone.copy()
    for m in c.mcs:
        if m["device"].endswith("ECMO"):
            paralysed_like[m["start"]:m["end"]] = True
    depth = (0.2 + 0.4 * np.clip(c.resp, 0, 1.2) / 1.2 + 0.3 * paralysed_like
             + 0.2 * (since_intub < 12) + _ou(rng, n, 0.08, 0.9))
    depth = np.where(sedated, np.clip(depth, 0, 1), 0.0)
    c.sed_depth = depth

    # Delirium: older, sicker, longer on the ventilator, benzodiazepines
    imv_days = np.cumsum(on_imv) / 24.0
    benzo = c.sedative == "midazolam"
    score = (-1.4 + 0.03 * (c.age - 65) + 1.4 * _lag(c.severity, 24.0) + 0.6 * benzo * sedated
             + 0.15 * np.minimum(imv_days, 5) + _ou(rng, n, 0.3, 0.97))
    delirium = np.zeros(n, dtype=bool)
    on = False
    for i in range(n):
        if not on and score[i] > 0.15:
            on = True
        elif on and score[i] < -0.15:
            on = False
        delirium[i] = on
    c.delirium = delirium
    hyperactive = rng.random() < 0.3

    enceph = np.clip(c.severity - 0.8, 0, None)
    rass = np.zeros(n)
    rass[sedated] = -0.3 - 4.4 * depth[sedated] + _ou(rng, n, 0.4, 0.8)[sedated]
    awake = ~sedated
    rass[awake] = (-3.5 * enceph[awake] + (1.4 * delirium[awake] if hyperactive else -0.8 * delirium[awake])
                   + _ou(rng, n, 0.35, 0.8)[awake])
    c.rass = np.clip(np.round(rass), -5, 4).astype(int)

    eye = np.select([c.rass <= -4, c.rass == -3, c.rass == -2, c.rass == -1], [1, 2, 3, 3], 4)
    motor = np.select([c.rass == -5, c.rass == -4, c.rass == -3, c.rass == -2], [2, 4, 5, 6], 6)
    verbal = np.select([c.rass <= -3, c.rass == -2, delirium], [1, 3, 4], 5)
    verbal = np.where(on_imv, 1, verbal)  # intubated or trached: verbal not testable
    c.gcs_eye, c.gcs_verbal, c.gcs_motor = eye.astype(int), verbal.astype(int), motor.astype(int)


# ----------------------------------------------------------------------
# Fluids, nutrition, urine output
# ----------------------------------------------------------------------
def _fluids(c, rng):
    n = c.n_hours
    boluses = []
    crystalloid = "Lactated Ringers" if rng.random() < 0.6 else "Normal Saline"
    triggers = [s for s, _ in _runs(c.shock > 0.45)] + [s for s, _ in _runs(c.ne_eq > 0)]
    done = -99
    for s in sorted(triggers):
        if s - done < 12:
            continue
        done = s
        target = 30 * c.weight_kg if c.infected else 1000
        given = 0.0
        h = max(s - rng.uniform(0, 1), 0)
        while given < target and h < n - 1:
            vol = 1000.0 if rng.random() < 0.6 else 500.0
            boluses.append((float(h), crystalloid, vol))
            given += vol
            h += rng.uniform(0.5, 1.5)
    c.fluid_boluses = boluses

    on_imv = c.level == 4
    npo = on_imv | (c.level == 3) | (c.severity > 0.7)
    feed = np.zeros(n)
    for s, e in c.imv_episodes:
        start = s + int(rng.uniform(24, 48))
        goal = float(rng.choice([45, 50, 55, 60, 65]))
        rate = 0.0
        for i in range(start, e):
            if (i - start) % 6 == 0:
                rate = min(rate + 10 + 10 * (rng.random() < 0.5), goal)
            if c.ne_eq[i] > 0.3:  # held on high-dose pressors
                rate = 0.0
            feed[i] = rate
    mivf = np.where(npo & (feed < 30), 75.0 if rng.random() < 0.6 else 100.0, 0.0)
    mivf = np.where(c.crrt, 0.0, mivf)
    c.mivf, c.tube_feed = mivf, feed

    hour_of_day = (c.admit_time.hour + np.arange(n)) % 24
    meals = []
    for i in range(n):
        if hour_of_day[i] in (8, 12, 18) and not npo[i] and feed[i] == 0:
            if rng.random() < 0.85:
                meals.append((float(i) + rng.uniform(0, 0.75), float(round(rng.uniform(150, 500), -1))))
    c.meals = meals

    base = 0.9 * c.weight_kg
    kidney = (1 - 0.9 * np.clip(c.aki, 0, 1)) * (1 - 0.45 * np.clip(c.shock, 0, 1))
    noise = np.exp(_ou(rng, n, 0.35, 0.6))
    uop = base * kidney * noise
    uop = np.where(c.crrt, uop * 0.2, uop)
    if c.comfort_h is not None:
        uop[c.comfort_h:] *= 0.3
    c.uop = np.clip(uop, 0, 600)


# ----------------------------------------------------------------------
# Code status
# ----------------------------------------------------------------------
def _code_status(c, rng):
    n = c.n_hours
    p_dnr = 0.3 if c.age >= 80 else 0.12 if c.age >= 70 else 0.04
    events = []
    if rng.random() < p_dnr:
        events.append((rng.uniform(0, 4), str(rng.choice(["DNR", "DNR/DNI"], p=[0.5, 0.5]))))
    else:
        if rng.random() < 0.3:
            events.append((rng.uniform(0, 1), "Presume Full"))
            events.append((rng.uniform(2, 12), "Full"))
        else:
            events.append((rng.uniform(0, 4), "Full"))
    current = events[-1][1]
    if c.expired:
        if c.comfort_h is not None:
            t = max(c.comfort_h - rng.uniform(1, 24), events[-1][0] + 0.5)
            new = "DNR" if c.level[min(int(t), n - 1)] == 4 else "DNR/DNI"
            if new != current:
                events.append((t, new))
        elif current in ("Full", "Presume Full") and rng.random() < 0.7:
            t = max(n - 2 - rng.uniform(6, 72), events[-1][0] + 0.5)
            events.append((t, "DNR"))
    elif current in ("Full", "Presume Full") and c.age >= 75 and c.severity.max() > 0.8 and rng.random() < 0.2:
        t = float(np.argmax(c.severity)) + rng.uniform(6, 48)
        if t < n - 2:
            events.append((t, "DNR"))
    c.code_status = [(float(min(t, n - 1.5)), cat) for t, cat in events]


# ----------------------------------------------------------------------
# Unit-level view of transfusions, shared by the transfusion and
# intake/output tables so both report the same products and volumes.
# ----------------------------------------------------------------------
_UNIT = {
    "Packed RBCs": ((280, 340), (1.5, 3.0)),
    "Fresh Frozen Plasma": ((220, 280), (0.5, 1.0)),
    "Platelets": ((200, 300), (0.5, 1.0)),
    "Cryoprecipitate": ((80, 120), (0.25, 0.5)),
}


def transfusion_units(c) -> list[dict]:
    """One record per unit: start/end hour, component, volume and attribute.

    Units of one order hang back to back. Products for a patient with marrow
    failure are irradiated; RBCs and platelets are otherwise leukoreduced.
    """
    cached = getattr(c, "_transfusion_units", None)
    if cached is not None:
        return cached
    rng = _stream(c, "transfusion_units")
    units = []
    busy_until = -1.0
    for t in c.transfusions:
        (v_lo, v_hi), (d_lo, d_hi) = _UNIT[t["component"]]
        start = max(t["h"], busy_until + rng.uniform(0.1, 0.4))
        for _ in range(int(t["units"])):
            end = start + rng.uniform(d_lo, d_hi)
            if end >= c.n_hours:
                break
            if c.marrow_failure:
                attr = "Irradiated"
            elif t["component"] in ("Packed RBCs", "Platelets"):
                attr = "Leukoreduced"
            else:
                attr = None
            units.append(dict(start_h=float(start), end_h=float(end), component=t["component"],
                              volume=float(round(rng.uniform(v_lo, v_hi), -1)), attribute=attr))
            busy_until = end
            start = end + rng.uniform(0.1, 0.4)
    units.sort(key=lambda u: u["start_h"])
    c._transfusion_units = units
    return units
