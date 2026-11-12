"""Helpers shared by the CLIF 3.0-only table generators."""

from __future__ import annotations

import hashlib
import zlib
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from synthetic_clif.physiology.course import HospitalCourse, course_for_row


def stable_id(*parts, n: int = 10) -> str:
    """Short, stable hex id from the given parts."""
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:n]


def stream(*parts) -> np.random.Generator:
    """Random stream keyed on the given parts (order-independent across stays)."""
    return np.random.default_rng(zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF)


def utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def iter_courses(hosps: pd.DataFrame) -> Iterator[tuple[dict, HospitalCourse, pd.Timestamp, pd.Timestamp]]:
    """(row, course, admission, discharge) for each stay with an admission time."""
    for _, row in hosps.iterrows():
        if pd.isna(row.get("admission_dttm")):
            continue
        course = course_for_row(row)
        if course is None:
            continue
        admit = utc(row["admission_dttm"])
        discharge = row.get("discharge_dttm")
        discharge = utc(discharge) if discharge is not None and not pd.isna(discharge) else \
            admit + pd.Timedelta(course.los_h, unit="h")
        yield dict(row), course, admit, discharge


def at(admit: pd.Timestamp, h: float) -> pd.Timestamp:
    return admit + pd.Timedelta(float(h), unit="h")


class StayLocations:
    """Where a patient was at a given time, from the ADT table."""

    def __init__(self, adt: Optional[pd.DataFrame]):
        self._by = {}
        if adt is None or adt.empty:
            return
        a = adt[["hospitalization_id", "in_dttm", "out_dttm", "location_category"]].copy()
        a["in_dttm"] = pd.to_datetime(a["in_dttm"], utc=True)
        a["out_dttm"] = pd.to_datetime(a["out_dttm"], utc=True)
        a["location_category"] = a["location_category"].astype(str).str.lower()
        for hid, g in a.sort_values("in_dttm").groupby("hospitalization_id", sort=False):
            self._by[hid] = list(zip(g["in_dttm"], g["out_dttm"], g["location_category"]))

    def at(self, hid, ts) -> Optional[str]:
        for a, b, loc in self._by.get(hid, []):
            if a <= ts and (pd.isna(b) or ts < b):
                return loc
        return None

    def segments(self, hid) -> list[tuple]:
        return self._by.get(hid, [])

    def icu_windows(self, hid, merge_gap_h: float = 6.0) -> list[tuple]:
        """Merged (in, out) windows spent in an ICU."""
        out = []
        for a, b, loc in self._by.get(hid, []):
            if loc != "icu":
                continue
            if out and pd.notna(out[-1][1]) and (a - out[-1][1]) <= pd.Timedelta(merge_gap_h, unit="h"):
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return out


def critical(course: HospitalCourse) -> np.ndarray:
    """Hours of intensive-care-level support (IMV/NIPPV, pressors, RRT)."""
    m = (course.level >= 3) | (course.ne_eq > 0) | course.crrt
    return np.asarray(m, dtype=bool)
