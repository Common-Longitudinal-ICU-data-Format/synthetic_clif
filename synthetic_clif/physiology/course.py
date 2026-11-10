"""Per-hospitalization latent clinical course.

Every time-series generator (vitals, respiratory support, continuous
medications, labs) used to draw its own values independently, so a patient
could be on high-dose norepinephrine with a normal MAP, on 100% FiO2 with an
SpO2 of 99, or have a creatinine that jumped at random from one day to the
next.

``HospitalCourse`` builds one hidden trajectory per hospitalization on an
hourly grid: an illness-severity curve, organ-specific scores (shock,
respiratory failure, kidney injury, infection), the device and ventilator
settings clinicians would choose in response, vasopressor titration, and the
resulting vital signs and lab values. The generators then *observe* this
course, adding only measurement noise, so values agree across tables.

The course is a deterministic function of the hospitalization row and a
course seed. The dataset orchestrator puts that seed in a ``_course_seed``
column (derived from the dataset seed); without it, the seed is a stable hash
of the hospitalization id, so generators used on their own still agree with
each other.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

# Device levels: 0 room air, 1 low-flow O2, 2 high-flow, 3 non-invasive PPV, 4 IMV
LEVEL_UP = np.array([0.20, 0.40, 0.55, 0.68])  # resp score needed to escalate to level k+1
LEVEL_DOWN = LEVEL_UP - 0.12                    # score below which a level can be weaned

IMV_MODES = [
    "Assist Control-Volume Control",
    "Pressure Control",
    "Pressure-Regulated Volume Control",
    "SIMV",
    "Pressure Support/CPAP",
]
IMV_MODE_WEIGHTS = np.array([0.597, 0.021, 0.168, 0.095, 0.063])
IMV_MODE_WEIGHTS = IMV_MODE_WEIGHTS / IMV_MODE_WEIGHTS.sum()

VENT_BRANDS = ["Puritan Bennett 840", "Servo-i", "Drager V500", "Hamilton G5", "Avea"]

PRESSOR_AGENTS = ["norepinephrine", "phenylephrine", "epinephrine", "dopamine"]
PRESSOR_WEIGHTS = np.array([0.65, 0.12, 0.12, 0.11])

# ARDSNet lower-PEEP/higher-FiO2 table (FiO2 -> PEEP choices)
_ARDSNET = [
    (0.30, [5]),
    (0.40, [5, 8]),
    (0.50, [8, 10]),
    (0.60, [10]),
    (0.70, [10, 12, 14]),
    (0.80, [14]),
    (0.90, [14, 16, 18]),
    (1.00, [18, 20]),
]


def _stable_hash(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def course_seed_for(dataset_seed: int, hospitalization_id) -> int:
    """Course seed for one hospitalization, derived from the dataset seed."""
    return _stable_hash(f"{int(dataset_seed)}:{hospitalization_id}")


def patient_traits(patient_seed: int, sex: str) -> dict:
    """Chronic traits that belong to the patient, not the stay.

    Height, body size and chronic disease are drawn from a patient-level
    seed so a readmitted patient keeps them across hospitalizations, and the
    patient-level diagnosis table can agree with them.
    """
    r = np.random.default_rng((int(patient_seed) + _stable_hash("patient")) & 0xFFFFFFFF)
    height = float(np.clip(r.normal(176 if sex == "male" else 163, 7.5), 140, 210))
    bmi = float(np.clip(r.normal(28, 6), 16, 55))
    diabetic = bool(r.random() < 0.25)
    copd = bool(r.random() < 0.18)
    ckd = bool(r.random() < 0.15)
    ckd_factor = float(r.uniform(1.6, 2.6)) if ckd else 1.0
    return dict(height=height, bmi=bmi, diabetic=diabetic, copd=copd, ckd=ckd, ckd_factor=ckd_factor)


def _ou(rng, n, sd, phi):
    """Stationary AR(1) noise on the hourly grid."""
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    innov = rng.normal(0, sd * math.sqrt(1 - phi**2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + innov[i]
    return x


def _lag(x, tau_h):
    """First-order lag (exponential smoothing) with time constant tau_h hours."""
    a = 1 - math.exp(-1.0 / tau_h)
    y = np.empty_like(x)
    y[0] = x[0] * 0.3
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def _q(x, step):
    return round(round(x / step) * step, 4)


def severinghaus_spo2(pao2):
    """SpO2 (%) from PaO2 (mmHg), Severinghaus 1979."""
    p = np.maximum(np.asarray(pao2, dtype=float), 1.0)
    return 100.0 / (1.0 + 23400.0 / (p**3 + 150.0 * p))


@dataclass
class HospitalCourse:
    """Hidden hourly trajectory for one hospitalization (index 0 = admission)."""

    hospitalization_id: str
    admit_time: pd.Timestamp
    n_hours: int
    rng_seed: int
    # Patient descriptors
    age: float
    sex: str
    height_cm: float
    weight_kg: float
    pbw_kg: float
    acuity: int
    # Latent hourly scores
    severity: np.ndarray = field(repr=False, default=None)
    shock: np.ndarray = field(repr=False, default=None)
    resp: np.ndarray = field(repr=False, default=None)
    aki: np.ndarray = field(repr=False, default=None)
    infection: np.ndarray = field(repr=False, default=None)
    # Respiratory support (hourly)
    level: np.ndarray = field(repr=False, default=None)
    device: list = field(repr=False, default=None)
    trach: np.ndarray = field(repr=False, default=None)
    mode: list = field(repr=False, default=None)
    fio2: np.ndarray = field(repr=False, default=None)        # delivered / set FiO2
    lpm: np.ndarray = field(repr=False, default=None)
    flow: np.ndarray = field(repr=False, default=None)
    peep: np.ndarray = field(repr=False, default=None)
    ps: np.ndarray = field(repr=False, default=None)
    pc: np.ndarray = field(repr=False, default=None)
    vt_set: np.ndarray = field(repr=False, default=None)
    rr_set: np.ndarray = field(repr=False, default=None)
    ti_set: np.ndarray = field(repr=False, default=None)
    crs: np.ndarray = field(repr=False, default=None)         # compliance mL/cmH2O
    raw: float = 10.0                                         # airway resistance
    vent_brand: Optional[str] = None
    imv_episodes: list = field(default_factory=list)          # [(start_h, end_h)]
    # Haemodynamics and other vitals (true values)
    pressor_agent: str = "norepinephrine"
    ne_eq: np.ndarray = field(repr=False, default=None)       # norepinephrine-equivalent mcg/kg/min
    vasopressin: np.ndarray = field(repr=False, default=None)  # bool
    dobutamine: np.ndarray = field(repr=False, default=None)   # bool
    map: np.ndarray = field(repr=False, default=None)
    sbp: np.ndarray = field(repr=False, default=None)
    dbp: np.ndarray = field(repr=False, default=None)
    hr: np.ndarray = field(repr=False, default=None)
    temp: np.ndarray = field(repr=False, default=None)
    rr: np.ndarray = field(repr=False, default=None)          # total observed rate
    pao2: np.ndarray = field(repr=False, default=None)
    spo2: np.ndarray = field(repr=False, default=None)
    # Medications
    sedative: str = "propofol"
    analgesic: str = "fentanyl"
    insulin: np.ndarray = field(repr=False, default=None)     # bool
    # Lab intercepts
    lab_base: dict = field(repr=False, default_factory=dict)
    _lab_daily: dict = field(repr=False, default_factory=dict)
    # Outcome and patient-level facts
    expired: bool = False
    infected: bool = False
    diabetic: bool = False
    copd: bool = False
    ckd: bool = False
    pf_true: np.ndarray = field(repr=False, default=None)     # PaO2/FiO2 before noise
    _map_u: np.ndarray = field(repr=False, default=None)      # MAP without pressors
    _response: float = 1.0
    # Clinical events built on top of the course (see physiology.events)
    infection_source: Optional[str] = None
    pathogen_gram: Optional[str] = None
    infection_recognized_h: Optional[float] = None
    cultures: list = field(repr=False, default_factory=list)
    antibiotics: list = field(repr=False, default_factory=list)
    bleed: np.ndarray = field(repr=False, default=None)       # cumulative Hgb loss g/dL
    crrt: np.ndarray = field(repr=False, default=None)        # bool
    crrt_mode: str = "cvvhdf"
    prone: np.ndarray = field(repr=False, default=None)       # bool
    mcs: list = field(repr=False, default_factory=list)       # [{device, start, end}]
    marrow_failure: bool = False
    los_h: float = 0.0                                         # admission to discharge, hours
    comfort_h: Optional[int] = None
    shock_type: str = "distributive"
    art_line: np.ndarray = field(repr=False, default=None)    # bool
    central_line: np.ndarray = field(repr=False, default=None)
    dialysis_line: np.ndarray = field(repr=False, default=None)
    pa_catheter: np.ndarray = field(repr=False, default=None)
    sed_depth: np.ndarray = field(repr=False, default=None)   # 0 light .. 1 deep
    rass: np.ndarray = field(repr=False, default=None)
    gcs_eye: np.ndarray = field(repr=False, default=None)
    gcs_verbal: np.ndarray = field(repr=False, default=None)
    gcs_motor: np.ndarray = field(repr=False, default=None)
    delirium: np.ndarray = field(repr=False, default=None)
    fluid_boluses: list = field(repr=False, default_factory=list)
    mivf: np.ndarray = field(repr=False, default=None)        # maintenance IV mL/h
    tube_feed: np.ndarray = field(repr=False, default=None)   # enteral feed mL/h
    meals: list = field(repr=False, default_factory=list)     # [(h, mL)]
    uop: np.ndarray = field(repr=False, default=None)         # urine mL/h
    transfusions: list = field(repr=False, default_factory=list)
    code_status: list = field(repr=False, default_factory=list)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        hospitalization_id,
        admit_time,
        discharge_time,
        seed: int,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        expired: bool = False,
        patient_seed: Optional[int] = None,
    ) -> "HospitalCourse":
        rng = np.random.default_rng(seed)
        admit_time = pd.Timestamp(admit_time)
        discharge_time = pd.Timestamp(discharge_time)
        los_h = max((discharge_time - admit_time).total_seconds() / 3600.0, 1.0)
        n = int(math.ceil(los_h)) + 2
        t = np.arange(n, dtype=float)

        if age is None or pd.isna(age):
            age = float(np.clip(rng.normal(62, 16), 18, 95))
        sex = str(sex).strip().lower() if sex is not None and not pd.isna(sex) else ""
        if sex not in ("male", "female"):
            sex = "male" if rng.random() < 0.55 else "female"
        traits = patient_traits(seed if patient_seed is None else patient_seed, sex)
        height = traits["height"]
        bmi = traits["bmi"] * math.exp(rng.normal(0, 0.03))  # weight drifts between stays
        weight = bmi * (height / 100) ** 2
        pbw = (50.0 if sex == "male" else 45.5) + 0.91 * (height - 152.4)
        pbw = max(pbw, 35.0)

        # Acuity has its own stream so the outcome model can look at it
        # without perturbing the rest of the course.
        acuity_rng = np.random.default_rng((int(seed) + _stable_hash("acuity")) & 0xFFFFFFFF)
        acuity = int(acuity_rng.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.35, 0.15]))
        c = cls(
            hospitalization_id=str(hospitalization_id),
            admit_time=admit_time,
            n_hours=n,
            rng_seed=int(seed),
            age=float(age),
            sex=sex,
            height_cm=round(height, 1),
            weight_kg=round(weight, 1),
            pbw_kg=pbw,
            acuity=acuity,
        )
        c.expired = bool(expired)
        c.los_h = float(los_h)
        c.diabetic, c.copd, c.ckd = traits["diabetic"], traits["copd"], traits["ckd"]
        c._ckd_factor = traits["ckd_factor"]
        c._build_severity(rng, t, los_h, expired)
        c._build_respiratory(rng)
        c._build_haemodynamics(rng)
        c._build_meds_and_labs(rng)
        from synthetic_clif.physiology.events import build_events

        build_events(c)
        return c

    def _build_severity(self, rng, t, los_h, expired):
        n = len(t)
        amp = {
            1: rng.uniform(0.05, 0.30),
            2: rng.uniform(0.25, 0.55),
            3: rng.uniform(0.50, 0.85),
            4: rng.uniform(0.80, 1.15),
        }[self.acuity]

        def episode(a, onset, ramp, plateau, tau):
            y = np.zeros(n)
            rising = (t >= onset) & (t < onset + ramp)
            y[rising] = (t[rising] - onset) / ramp
            flat = (t >= onset + ramp) & (t < onset + ramp + plateau)
            y[flat] = 1.0
            after = t >= onset + ramp + plateau
            y[after] = np.exp(-(t[after] - onset - ramp - plateau) / tau)
            return a * y

        # Most patients arrive already ill (onset before admission)
        onset = rng.uniform(-12, 2) if rng.random() < 0.75 else rng.uniform(0, 0.5 * los_h)
        sev = episode(
            amp,
            onset,
            rng.uniform(2, 12),
            rng.uniform(4, 24 + 60 * amp),
            rng.uniform(18, 72) * (0.6 + amp),
        )
        # Smouldering baseline illness keeps some patients mildly abnormal
        sev = sev + 0.1 * amp
        if self.acuity >= 2 and rng.random() < 0.15 and los_h > 72:
            onset2 = rng.uniform(0.3, 0.8) * los_h
            sev = np.maximum(
                sev,
                episode(rng.uniform(0.5, 1.0) * max(amp, 0.5), onset2,
                        rng.uniform(2, 8), rng.uniform(6, 36), rng.uniform(18, 60)),
            )
        if expired:
            # Separate stream: a terminal decline must not reshuffle the
            # earlier course the outcome model was judged on.
            erng = np.random.default_rng((self.rng_seed + _stable_hash("expiry")) & 0xFFFFFFFF)
            span = min(erng.uniform(12, 72), los_h)
            ramp = np.clip((t - (los_h - span)) / span, 0, 1)
            sev = np.maximum(sev, 1.2 * ramp)

        w = rng.uniform(0.55, 1.05, 4)  # shock, resp, aki, infection
        dominant = rng.choice(4, p=[0.30, 0.50, 0.10, 0.10])
        w[dominant] *= 1.3
        others = [i for i in range(4) if i != dominant]
        w[others] *= rng.uniform(0.35, 0.95, 3)

        self.severity = np.clip(sev, 0, 1.4)
        self.shock = np.clip(w[0] * sev + _ou(rng, n, 0.05, 0.9), 0, 1.6)
        self.resp = np.clip(w[1] * sev + _ou(rng, n, 0.05, 0.9), 0, 1.6)
        self.aki = np.clip(_lag(w[2] * sev, 24.0), 0, 1.5)
        infected = rng.random() < 0.55
        self.infected = bool(infected)
        self.infection = np.clip(_lag(w[3] * sev, 8.0) * (1.0 if infected else 0.25), 0, 1.5)

    def _build_respiratory(self, rng):
        n = self.n_hours
        r = self.resp + 0.6 * np.maximum(0.0, self.shock - 0.9)
        # PaO2/FiO2 ratio falls with respiratory failure
        pf = 450.0 * np.exp(-1.4 * np.clip(r, 0, 1.4))
        fio2_need = np.clip(70.0 / pf, 0.21, 1.0)

        prefers_mask = rng.random() < 0.15
        prefers_cpap = rng.random() < 0.20
        will_trach = rng.random() < 0.4
        peep_minimal = rng.random() < 0.7
        self.vent_brand = str(rng.choice(VENT_BRANDS))
        self.raw = float(np.clip(rng.normal(10, 3), 5, 20))
        crs0 = float(np.clip(rng.normal(50, 10), 25, 80))

        level = np.zeros(n, dtype=int)
        trach = np.zeros(n, dtype=int)
        lvl = int((r[0] > LEVEL_UP).sum())
        since_change = 0
        imv_start = None
        trach_at = None
        episodes = []
        if lvl == 4:  # intubated before arrival
            imv_start = 0
            trach_at = int(rng.uniform(168, 336)) if will_trach else None
        for i in range(n):
            up = int((r[i] > LEVEL_UP).sum())
            down = int((r[i] > LEVEL_DOWN).sum())
            new = lvl
            if since_change >= 2:
                if up > lvl:
                    new = up
                elif lvl == 4:
                    on_for = i - imv_start
                    calm = i >= 6 and (r[i - 6:i + 1] < 0.5).all() and (self.shock[i - 6:i + 1] < 0.5).all()
                    if on_for >= 12 and calm:
                        new = max(min(down, 2), 1)
                elif down < lvl:
                    new = lvl - 1
            if new == 4 and lvl != 4:
                imv_start = i
                trach_at = (i + int(rng.uniform(168, 336))) if will_trach else None
            if lvl == 4 and new != 4:
                episodes.append((imv_start, i))
                imv_start = None
            if new != lvl:
                since_change = 0
            lvl = new
            since_change += 1
            if lvl == 4 and trach_at is not None and i >= trach_at:
                trach[i:] = 1
                trach_at = None
            level[i] = lvl
        if lvl == 4:
            episodes.append((imv_start, n))
        self.level = level
        self.trach = trach
        self.imv_episodes = episodes

        device, mode = [], []
        fio2 = np.full(n, 0.21)
        lpm = np.full(n, np.nan)
        flow = np.full(n, np.nan)
        peep = np.full(n, np.nan)
        ps = np.full(n, np.nan)
        pc = np.full(n, np.nan)
        vt_set = np.full(n, np.nan)
        rr_set = np.full(n, np.nan)
        ti_set = np.full(n, np.nan)
        crs = crs0 * (1 - 0.4 * np.clip(r, 0, 1.2))

        # Per-IMV-episode ventilator plan
        ep_plan = {}
        for (s, e) in episodes:
            m = str(rng.choice(IMV_MODES, p=IMV_MODE_WEIGHTS))
            wean_switch = m != "Pressure Support/CPAP" and (e - s) > 24 and rng.random() < 0.5
            wean_from = s + int(0.75 * (e - s)) if wean_switch else None
            vt = _q(self.pbw_kg * rng.uniform(6, 8), 10)
            ep_plan[s] = dict(
                end=e,
                mode=m,
                wean_from=wean_from,
                vt=vt,
                rr_base=float(rng.integers(14, 21)),
                ti=_q(rng.uniform(0.8, 1.2), 0.05),
                pc=float(np.clip(round(vt / max(crs[s], 15)), 10, 30)),
                ps=float(rng.choice([5, 8, 10, 12, 15], p=[0.25, 0.3, 0.25, 0.1, 0.1])),
            )

        cur_fio2 = 0.21
        cur_peep = 5.0
        cur_rr = None
        plan = None
        epap = float(rng.choice([5, 6, 8, 10], p=[0.35, 0.2, 0.3, 0.15]))
        ipap_ps = float(rng.choice([5, 8, 10, 12], p=[0.2, 0.35, 0.3, 0.15]))
        last_fio2_change = -99
        for i in range(n):
            lv = level[i]
            need = fio2_need[i]
            if lv == 4:
                if i in ep_plan:
                    plan = ep_plan[i]
                    cur_fio2 = 1.0            # intubated on 100%, then weaned
                    cur_rr = plan["rr_base"]
                    last_fio2_change = i
                m = plan["mode"]
                if plan["wean_from"] is not None and i >= plan["wean_from"]:
                    m = "Pressure Support/CPAP"
                mode.append(m)
                device.append("IMV")
                target = float(np.clip(_q(need / (1 + 0.02 * max(cur_peep - 5, 0)) + 0.03, 0.05), 0.30, 1.0))
                if abs(target - cur_fio2) >= 0.1 or (
                    target < cur_fio2 and i - last_fio2_change >= 2 and rng.random() < 0.5
                ):
                    step = 0.1 if target < cur_fio2 else (target - cur_fio2)
                    cur_fio2 = float(np.clip(_q(cur_fio2 - step if target < cur_fio2 else cur_fio2 + step, 0.05), 0.30, 1.0))
                    last_fio2_change = i
                fio2[i] = cur_fio2
                table = next(p for f, p in _ARDSNET if cur_fio2 <= f + 1e-9)
                target_peep = float(min(table) if peep_minimal else table[len(table) // 2])
                if peep_minimal:
                    target_peep = max(5.0, target_peep - 3)
                if abs(target_peep - cur_peep) >= 2 or i in ep_plan:
                    cur_peep = target_peep
                peep[i] = cur_peep
                desired_rr = float(np.clip(plan["rr_base"] + round(5 * min(self.shock[i], 1.2)), 10, 30))
                if abs(desired_rr - cur_rr) >= 3:
                    cur_rr = desired_rr
                if m == "Pressure Support/CPAP":
                    ps[i] = plan["ps"]
                else:
                    rr_set[i] = cur_rr
                    if m == "Pressure Control":
                        pc[i] = plan["pc"]
                        ti_set[i] = plan["ti"]
                    else:
                        vt_set[i] = plan["vt"]
                        if m == "Pressure-Regulated Volume Control":
                            ti_set[i] = plan["ti"]
                    if m == "SIMV":
                        ps[i] = plan["ps"]
                continue

            mode.append(None)
            plan = None
            if self.trach[i]:
                device.append("Trach Collar")
                fio2[i] = float(np.clip(_q(need + 0.05, 0.05), 0.28, 0.5))
                lpm[i] = float(np.clip(round(5 + 20 * (fio2[i] - 0.28)), 5, 10))
                continue
            if lv == 0:
                device.append("Room Air")
                fio2[i] = 0.21
            elif lv == 1:
                if prefers_mask:
                    device.append("Face Mask")
                    lpm[i] = float(np.clip(round(6 + (need - 0.3) * 40), 6, 15))
                    fio2[i] = float(np.clip(_q(0.3 + 0.02 * (lpm[i] - 6), 0.01), 0.28, 0.5))
                else:
                    device.append("Nasal Cannula")
                    lpm[i] = float(np.clip(math.ceil((need - 0.21) / 0.04), 1, 6))
                    fio2[i] = round(0.21 + 0.04 * lpm[i], 2)
            elif lv == 2:
                device.append("High Flow NC")
                fio2[i] = float(np.clip(_q(need + 0.05, 0.05), 0.30, 1.0))
                flow[i] = float(np.clip(_q(30 + 30 * r[i], 5), 30, 60))
            else:
                device.append("CPAP" if prefers_cpap else "NIPPV")
                fio2[i] = float(np.clip(_q(need + 0.05, 0.05), 0.30, 1.0))
                peep[i] = epap
                if not prefers_cpap:
                    ps[i] = ipap_ps
            cur_fio2 = fio2[i]

        self.device = device
        self.mode = mode
        self.fio2 = fio2
        self.lpm = lpm
        self.flow = flow
        self.peep = peep
        self.ps = ps
        self.pc = pc
        self.vt_set = vt_set
        self.rr_set = rr_set
        self.ti_set = ti_set
        self.crs = crs

        peep_gain = 1 + 0.02 * np.nan_to_num(np.maximum(peep - 5, 0))
        self.pf_true = pf * peep_gain
        self.pao2 = np.clip(pf * peep_gain * fio2 * np.exp(_ou(rng, n, 0.12, 0.7)), 35, 550)
        self.spo2 = np.clip(severinghaus_spo2(self.pao2), 60, 100)

    def _build_haemodynamics(self, rng):
        n = self.n_hours
        age = self.age
        map0 = rng.normal(88, 8) + 0.08 * (age - 60)
        pp0 = float(np.clip(rng.normal(46, 9) + 0.35 * (age - 60), 25, 95))
        hr0 = rng.normal(82, 10)
        temp0 = rng.normal(36.9, 0.25)
        rr0 = rng.normal(16, 2)
        on_imv = self.level == 4
        sedated = on_imv & (self.trach == 0)

        map_u = map0 - 40 * self.shock - 5 * sedated + _ou(rng, n, 3.0, 0.8)

        self.pressor_agent = str(rng.choice(PRESSOR_AGENTS, p=PRESSOR_WEIGHTS))
        vaso_ok = rng.random() < 0.6
        dobu_ok = rng.random() < 0.07
        ne = np.zeros(n)
        vaso = np.zeros(n, dtype=bool)
        dobu = np.zeros(n, dtype=bool)
        map_goal = rng.uniform(65, 72)        # team's MAP target
        response = rng.uniform(0.7, 1.3)      # mmHg gained per unit dose, relative
        on = False
        dose = 0.0
        v_on = False
        low_streak = 0
        prev_map = map_u[0]
        for i in range(n):
            low_streak = low_streak + 1 if map_u[i] < 61 else 0
            if not on and low_streak >= 2 and self.shock[i] > 0.4:
                on = True
                dose = float(np.clip(0.05 + rng.uniform(0, 0.1), 0.02, 0.3))
            elif on and rng.random() < 0.6:
                # titrate on the MAP seen last hour, in discrete steps
                if prev_map < map_goal - 2:
                    dose = min(dose + rng.choice([0.02, 0.04, 0.06]), 1.2)
                elif prev_map > map_goal + 5:
                    dose = max(dose - rng.choice([0.02, 0.03, 0.05]), 0.0)
                if map_u[i] > 70 and dose <= 0.03:
                    on, dose = False, 0.0
            if on and vaso_ok and dose > 0.25:
                v_on = True
            if v_on and (not on or dose < 0.1):
                v_on = False
            ne[i] = round(dose, 3) if on else 0.0
            vaso[i] = v_on
            dobu[i] = on and dobu_ok
            prev_map = map_u[i] + response * (60 * ne[i] + 6 * v_on)
        self.ne_eq = ne
        self.vasopressin = vaso
        self.dobutamine = dobu

        self._map_u = map_u
        self._response = float(response)
        self.map = map_u + response * (60 * ne + 6 * vaso)
        pp = pp0 * (1 - 0.25 * np.clip(self.shock, 0, 1.2)) + _ou(rng, n, 3.0, 0.8)
        pp = np.clip(pp, 15, 110)
        self.dbp = self.map - pp / 3
        self.sbp = self.map + 2 * pp / 3

        hypothermic = rng.random() < 0.1
        fever = (-1.5 if hypothermic else 1.8) * self.infection
        self.temp = temp0 + fever + _ou(rng, n, 0.2, 0.85)
        inotrope = self.pressor_agent in ("epinephrine", "dopamine")
        self.hr = (
            hr0 + 30 * self.shock + 10 * (self.temp - 37) + 8 * self.resp
            + 8 * inotrope * (ne > 0) + 6 * dobu + _ou(rng, n, 5.0, 0.85)
        )
        rr_spont = rr0 + 12 * self.resp + 4 * self.shock + _ou(rng, n, 2.0, 0.8)
        rr = rr_spont.copy()
        spont_mode = np.array([m == "Pressure Support/CPAP" for m in self.mode])
        set_rate = on_imv & ~spont_mode & ~np.isnan(self.rr_set)
        excess = np.maximum(0.0, 0.4 * (rr_spont - np.nan_to_num(self.rr_set)))
        excess = np.where(self.severity > 0.7, 0.0, excess)  # deeply sedated: no triggering
        rr[set_rate] = self.rr_set[set_rate] + excess[set_rate]
        self.rr = np.clip(rr, 6, 45)

    def _build_meds_and_labs(self, rng):
        n = self.n_hours
        self.sedative = str(rng.choice(["propofol", "dexmedetomidine", "midazolam"], p=[0.6, 0.3, 0.1]))
        self.analgesic = str(rng.choice(["fentanyl", "morphine"], p=[0.75, 0.25]))
        male = self.sex == "male"
        diabetic = self.diabetic
        b = {
            "sodium": rng.normal(139, 2.5),
            "potassium": rng.normal(4.1, 0.35),
            "bicarbonate": rng.normal(24.5, 2.0),
            "anion_gap": rng.normal(12.5, 2.0),
            "creatinine": math.exp(rng.normal(math.log(0.85 if not male else 1.0), 0.25)) * self._ckd_factor,
            "bun_ratio": math.exp(rng.normal(math.log(15), 0.3)),
            "glucose_serum": rng.normal(150 if diabetic else 108, 25 if diabetic else 12),
            "calcium_corr": rng.normal(9.3, 0.4),
            "magnesium": rng.normal(2.0, 0.2),
            "phosphate": rng.normal(3.5, 0.5),
            "albumin": rng.normal(3.7, 0.4),
            "total_protein": rng.normal(6.9, 0.5),
            "ast": math.exp(rng.normal(math.log(24), 0.35)),
            "alt": math.exp(rng.normal(math.log(22), 0.45)),
            "alkaline_phosphatase": math.exp(rng.normal(math.log(80), 0.3)),
            "bilirubin_total": math.exp(rng.normal(math.log(0.6), 0.4)),
            "lactate": math.exp(rng.normal(math.log(1.1), 0.25)),
            "hemoglobin": rng.normal(12.3 if male else 11.3, 1.7),
            "wbc": math.exp(rng.normal(math.log(7.5), 0.3)),
            "platelet_count": float(np.clip(math.exp(rng.normal(math.log(220), 0.35)), 40, 600)),
            "inr": rng.normal(1.05, 0.08),
            "ptt": rng.normal(30, 3),
            "crp": math.exp(rng.normal(math.log(4), 1.0)),
            "procalcitonin": math.exp(rng.normal(math.log(0.08), 0.8)),
            "troponin_i": math.exp(rng.normal(math.log(0.01), 0.9)),
            "pco2_offset": rng.normal(0, 2.5),
            "hypercapnic": float(self.copd),
            "scvo2": rng.normal(72, 4),
        }
        self.marrow_failure = bool(rng.random() < 0.06)
        if self.marrow_failure:  # e.g. haematologic malignancy
            b["platelet_count"] = float(rng.uniform(15, 60))
            b["hemoglobin"] -= 1.5
        self.lab_base = b
        glucose = b["glucose_serum"] + 45 * self.severity
        ins = np.zeros(n, dtype=bool)
        on = False
        for i in range(n):
            if not on and glucose[i] > 180:
                on = True
            elif on and glucose[i] < 140:
                on = False
            ins[i] = on
        self.insulin = ins

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def hour_index(self, ts) -> int:
        h = (pd.Timestamp(ts) - self.admit_time).total_seconds() / 3600.0
        return int(np.clip(math.floor(h), 0, self.n_hours - 1))

    def at(self, name: str, ts):
        """Value of an hourly array (linear interpolation for numeric arrays)."""
        arr = getattr(self, name)
        h = (pd.Timestamp(ts) - self.admit_time).total_seconds() / 3600.0
        h = float(np.clip(h, 0, self.n_hours - 1))
        if isinstance(arr, list) or arr.dtype.kind in "biu":
            return arr[int(h)]
        lo = int(h)
        hi = min(lo + 1, self.n_hours - 1)
        f = h - lo
        a, b = arr[lo], arr[hi]
        if np.isnan(a) or np.isnan(b):
            return a
        return a + f * (b - a)

    def _daily(self, key: str, sd: float, phi: float = 0.6) -> np.ndarray:
        """Daily AR(1) deviation for one lab (persists across draws)."""
        if key not in self._lab_daily:
            n_days = self.n_hours // 24 + 3
            r = np.random.default_rng((self.rng_seed + _stable_hash(key)) & 0xFFFFFFFF)
            self._lab_daily[key] = _ou(r, n_days, sd, phi)
        return self._lab_daily[key]

    def _d(self, key: str, h: float, sd: float, phi: float = 0.6) -> float:
        arr = self._daily(key, sd, phi)
        d = h / 24.0
        lo = int(d)
        hi = min(lo + 1, len(arr) - 1)
        return float(arr[lo] + (d - lo) * (arr[hi] - arr[lo]))

    # ------------------------------------------------------------------
    # True values shared by the event builder and the lab draws
    # ------------------------------------------------------------------
    def _i(self, h: float) -> int:
        return int(np.clip(h, 0, self.n_hours - 1))

    def _hgb_raw(self, h: float) -> float:
        """Haemoglobin before transfusion and measurement noise."""
        i = self._i(h)
        b = self.lab_base
        v = (b["hemoglobin"] - 2.2 * min(self.severity[i], 1.0) - 0.15 * min(h / 24.0, 14)
             + self._d("hemoglobin", h, 0.3, 0.8))
        if self.bleed is not None:
            v -= self.bleed[i]
        return v

    def _plt_raw(self, h: float) -> float:
        i = self._i(h)
        b = self.lab_base
        shock_lag = float(_lag_value(self.shock, i, 24.0))
        return (b["platelet_count"]
                * max(1 - 0.75 * min(_lag_value(self.infection, i, 24.0), 1.0) - 0.25 * min(shock_lag, 1.0), 0.08)
                * math.exp(self._d("plt", h, 0.06)))

    def _inr_raw(self, h: float) -> float:
        i = self._i(h)
        shock_lag = float(_lag_value(self.shock, i, 12.0))
        return self.lab_base["inr"] + 0.9 * max(shock_lag - 0.2, 0) ** 1.5 + self._d("inr", h, 0.04)

    def product_effect(self, h: float, component: str, per_unit: float, tau_h: Optional[float]) -> float:
        """Lab change from blood products given before hour ``h``."""
        tot = 0.0
        for tr in self.transfusions:
            if tr["component"] != component or tr["h"] + 1.0 > h:
                continue
            decay = 1.0 if tau_h is None else math.exp(-(h - tr["h"]) / tau_h)
            tot += per_unit * tr["units"] * decay
        return tot

    def crrt_effect(self, i: int) -> float:
        """Fraction of solute clearance achieved by CRRT at hour i (0..~1)."""
        eff = getattr(self, "_crrt_eff", None)
        return 0.0 if eff is None else float(eff[self._i(i)])

    def svo2_true(self, i: int) -> float:
        i = self._i(i)
        s = min(self.shock[i], 1.2)
        cardiogenic = self.shock_type == "cardiogenic"
        return self.lab_base["scvo2"] - 15 * s - 6 * cardiogenic * s

    def hemodynamics_at(self, ts, rng) -> dict:
        """Pulmonary-artery catheter readings that agree with MAP and shock type."""
        h = float(np.clip((pd.Timestamp(ts) - self.admit_time).total_seconds() / 3600.0, 0, self.n_hours - 1))
        i = int(h)
        s = min(self.shock[i], 1.4)
        bsa = math.sqrt(self.height_cm * self.weight_kg / 3600.0)
        cardiogenic = self.shock_type == "cardiogenic"
        if cardiogenic:
            ci = 2.8 - 1.3 * min(s, 1.2) + 0.4 * bool(self.dobutamine[i])
        elif self.shock_type == "hypovolemic":
            ci = 2.9 - 0.9 * s
        else:
            ci = 3.0 + 1.3 * s
        ci = max(ci + rng.normal(0, 0.15), 1.0)
        co = ci * bsa
        peep = 0.0 if np.isnan(self.peep[i]) else float(self.peep[i])
        cvp = 7 + 0.3 * max(peep - 5, 0) + 6 * cardiogenic * s + rng.normal(0, 1.5)
        cvp = float(np.clip(cvp, 1, 25))
        pcwp = 10 + 12 * cardiogenic * s + 2 * min(self.resp[i], 1.0) + rng.normal(0, 1.5)
        pcwp = float(np.clip(pcwp, 4, 35))
        pvr_wood = 1.3 + 1.6 * min(self.resp[i], 1.2) + rng.normal(0, 0.2)
        mpap = pcwp + max(pvr_wood, 0.5) * co
        padp = pcwp + 2 + rng.normal(0, 1)
        pasp = 3 * mpap - 2 * padp
        mean_ap = float(self.map[i])
        svr = 80 * (mean_ap - cvp) / co
        return {
            "CVP": cvp,
            "PA Systolic": pasp,
            "PA Diastolic": padp,
            "PA Mean": mpap,
            "PCWP": pcwp,
            "Cardiac Output": co,
            "Cardiac Index": co / bsa,
            "SVR": svr,
            "PVR": 80 * (mpap - pcwp) / co,
            "SvO2": float(np.clip(self.svo2_true(i) + rng.normal(0, 1.5), 30, 90)),
        }

    def labs_at(self, ts, categories, rng) -> dict:
        """Measured lab values for a panel drawn at ``ts``.

        Values are the course's true values plus small analytic noise.
        Derived values (chloride, BUN, pH, PT) are computed from the other
        measured values in the same draw so a panel is internally consistent.
        """
        h = float(np.clip((pd.Timestamp(ts) - self.admit_time).total_seconds() / 3600.0, 0, self.n_hours - 1))
        i = int(h)
        b = self.lab_base
        S = self.severity[i]
        shock = self.shock[i]
        aki = self.aki[i]
        inf = self.infection[i]
        shock_lag = float(_lag_value(self.shock, i, 12.0))
        days = h / 24.0
        mul = lambda key, sd: math.exp(self._d(key, h, sd))
        noise = lambda sd: rng.normal(0, sd)

        out = {}
        lactate = b["lactate"] * mul("lactate", 0.15) + 7.0 * shock**1.8 + noise(0.1)
        lactate = max(lactate, 0.4)
        dial = self.crrt_effect(i)  # CRRT clears solutes the injured kidney cannot
        cr = b["creatinine"] * (1 + 2.5 * aki) * (1 - 0.55 * dial) * mul("creatinine", 0.06) * (1 + noise(0.03))
        bun = cr * b["bun_ratio"] * (1 + 0.3 * S) * (1 - 0.2 * dial) * mul("bun", 0.08) * (1 + noise(0.04))
        k = b["potassium"] + 0.7 * aki + self._d("potassium", h, 0.25)
        k = k - 0.8 * dial * (k - 4.1) + noise(0.1)
        hco3 = b["bicarbonate"] - 1.1 * (lactate - 1.0) - 3.0 * aki + self._d("bicarbonate", h, 1.0)
        hco3 = hco3 + 0.7 * dial * (24.0 - hco3) + noise(0.6)
        hco3 = float(np.clip(hco3, 6, 42))
        na = b["sodium"] + self._d("sodium", h, 1.6) + noise(0.7)
        ag = b["anion_gap"] + 0.9 * (lactate - 1.0) + 2.0 * aki
        cl = na - hco3 - ag + noise(0.8)
        glucose = b["glucose_serum"] + 45 * S + self._d("glucose", h, 18, 0.3) + noise(6)
        if self.insulin[i]:
            glucose -= 30
        alb = b["albumin"] - 0.7 * min(S, 1.0) - 0.02 * days + self._d("albumin", h, 0.1) + noise(0.05)
        alb = max(alb, 1.5)
        ca = b["calcium_corr"] - 0.8 * (4.0 - alb) + self._d("calcium", h, 0.2) + noise(0.1)
        hgb = self._hgb_raw(h) + self.product_effect(h, "Packed RBCs", 1.0, None) + noise(0.15)
        hgb = max(hgb, 4.0)
        wbc = b["wbc"] * (1 + 1.4 * inf) * mul("wbc", 0.15) * (1 + noise(0.04))
        plt = (self._plt_raw(h) + self.product_effect(h, "Platelets", 30.0, 72.0)) * (1 + noise(0.04))
        inr = self._inr_raw(h) - self.product_effect(h, "Fresh Frozen Plasma", 0.25, 12.0) + noise(0.02)
        inr = max(inr, 0.85)
        ast = b["ast"] * (1 + 25 * max(shock_lag - 0.6, 0) ** 1.5 + 0.5 * S) * mul("ast", 0.15) * (1 + noise(0.05))
        alt = b["alt"] * (1 + 18 * max(shock_lag - 0.6, 0) ** 1.5 + 0.4 * S) * mul("alt", 0.15) * (1 + noise(0.05))
        pco2 = 40 + 1.2 * (hco3 - 24) + b["pco2_offset"] + 15 * b["hypercapnic"] * min(self.resp[i], 1.0) + noise(1.5)
        pco2 = float(np.clip(pco2, 18, 110))
        values = {
            "sodium": na,
            "potassium": k,
            "chloride": cl,
            "bicarbonate": hco3,
            "bun": bun,
            "creatinine": cr,
            "glucose_serum": glucose,
            "calcium_total": ca,
            "magnesium": b["magnesium"] + 0.1 * aki + self._d("magnesium", h, 0.12) + noise(0.05),
            "phosphate": b["phosphate"] + 1.2 * aki * (1 - 0.7 * dial) + self._d("phosphate", h, 0.4) + noise(0.15),
            "albumin": alb,
            "total_protein": b["total_protein"] - 0.9 * min(S, 1.0) + self._d("tp", h, 0.15) + noise(0.1),
            "ast": ast,
            "alt": alt,
            "alkaline_phosphatase": b["alkaline_phosphatase"] * (1 + 0.3 * S) * mul("alp", 0.08),
            "bilirubin_total": b["bilirubin_total"] * (1 + 1.5 * _lag_value(self.severity, i, 24.0)) * mul("bili", 0.1),
            "lactate": lactate,
            "hemoglobin": hgb,
            "wbc": wbc,
            "platelet_count": plt,
            "inr": inr,
            "pt": 11.5 * inr + noise(0.3),
            "ptt": b["ptt"] + 10 * shock_lag + self._d("ptt", h, 2.0) + noise(1.0),
            "crp": b["crp"] + 180 * _lag_value(self.infection, i, 12.0) * mul("crp", 0.1),
            "procalcitonin": b["procalcitonin"] * math.exp(3.5 * min(inf, 1.2)) * mul("pct", 0.15),
            "troponin_i": b["troponin_i"] * (1 + 40 * shock**2) * mul("trop", 0.2),
            "pco2_arterial": pco2,
            "po2_arterial": self.pao2[i] * (1 + noise(0.04)),
            "pco2_venous": pco2 + 6 + noise(1.5),
            "so2_central_venous": self.svo2_true(i) + 3 + noise(2.0),
        }
        values["bilirubin_conjugated"] = values["bilirubin_total"] * 0.3 * (1 + noise(0.05))
        values["ph_arterial"] = 6.1 + math.log10(hco3 / (0.0307 * pco2))
        values["ph_venous"] = 6.1 + math.log10(hco3 / (0.0307 * values["pco2_venous"]))
        for cat in categories:
            if cat in values:
                out[cat] = float(values[cat])
        return out


def _lag_value(arr: np.ndarray, i: int, tau_h: float) -> float:
    """Value at index i of a first-order lag of arr (computed on a window)."""
    start = max(0, i - int(5 * tau_h))
    a = 1 - math.exp(-1.0 / tau_h)
    y = arr[start] * (0.3 if start == 0 else 1.0)
    for v in arr[start + 1:i + 1]:
        y += a * (v - y)
    return float(y)


@lru_cache(maxsize=4096)
def _cached_course(hosp_id, admit, discharge, seed, age, sex, expired, patient_seed):
    return HospitalCourse.build(hosp_id, admit, discharge, seed, age=age, sex=sex, expired=expired,
                                patient_seed=patient_seed)


def patient_seed_for_row(row) -> Optional[int]:
    seed = row.get("_patient_seed")
    if seed is not None and not pd.isna(seed):
        return int(seed)
    pid = row.get("patient_id")
    if pid is not None and not pd.isna(pid):
        return _stable_hash(f"patient:{pid}")
    return None


def course_for_row(row) -> Optional[HospitalCourse]:
    """Course for a hospitalization row (Series or dict). None if no admission time."""
    admit = row.get("admission_dttm")
    if admit is None or pd.isna(admit):
        return None
    discharge = row.get("discharge_dttm")
    if discharge is None or pd.isna(discharge):
        discharge = pd.Timestamp(admit) + pd.Timedelta(days=5)
    hosp_id = row.get("hospitalization_id")
    seed = row.get("_course_seed")
    if seed is None or pd.isna(seed):
        seed = _stable_hash(f"course:{hosp_id}")
    age = row.get("age_at_admission")
    age = None if age is None or pd.isna(age) else float(age)
    sex = row.get("_sex")
    sex = None if sex is None or pd.isna(sex) else str(sex)
    dc = str(row.get("discharge_category") or "").strip().lower()
    return _cached_course(
        str(hosp_id), pd.Timestamp(admit), pd.Timestamp(discharge), int(seed), age, sex, dc == "expired",
        patient_seed_for_row(row),
    )
