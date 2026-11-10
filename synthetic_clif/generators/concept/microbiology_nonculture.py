"""Microbiology non-culture (PCR) test generator.

Tests follow the hospital course:

- respiratory viral PCR at presentation, nearly always for a patient with
  respiratory illness or a lung infection and less often as screening; a
  virus is found mainly in lung infections
- C. difficile toxin PCR later in the stay, mostly after several days of
  antibiotics, which is also when it turns positive

CLIF 2.1's non-culture vocabulary has no influenza, so influenza results are
only produced when ``include_influenza`` is set (the CLIF 3.0 path).
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row

LOINC_CODES = {
    "COVID-19 PCR": "94500-6",
    "Influenza A/B PCR": "92142-9",
    "RSV PCR": "92131-2",
    "C. diff Toxin PCR": "34713-8",
}

# test -> [(organism_category, organism_group)]
TESTS = {
    "COVID-19 PCR": [("sars_cov2", "viral_other")],
    "Influenza A/B PCR": [("influenza_a", "influenza"), ("influenza_b", "influenza")],
    "RSV PCR": [("respiratory_syncytial_virus", "respiratory_syncytial_virus")],
    "C. diff Toxin PCR": [("clostridium_difficile", "clostridium_difficile")],
}


class MicrobiologyNoncultureGenerator(BaseGenerator):
    """Generate PCR results from the shared course."""

    COLUMNS = ["patient_id", "hospitalization_id", "order_dttm", "collect_dttm", "result_dttm",
               "fluid_category", "fluid_name", "method_category", "method_name", "organism_category",
               "organism_group", "result_category", "result_name", "micro_order_name",
               "reference_low", "reference_high", "result_units", "lab_loinc_code"]

    def generate(self, hospitalizations_df: pd.DataFrame, include_influenza: bool = False,
                 **kwargs) -> pd.DataFrame:
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            c = course_for_row(hosp)
            if c is None:
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            for test, h, organism, group, positive, fluid in self._tests(c, include_influenza):
                order = admit + timedelta(hours=h)
                collect = order + timedelta(minutes=float(self.rng.uniform(10, 60)))
                result = collect + timedelta(hours=float(self.rng.uniform(1, 8)))
                if collect >= discharge:
                    continue
                cat = "detected" if positive else "not_detected"
                if not positive and self.rng.random() < 0.005:
                    cat = "indeterminate"
                records.append({
                    "patient_id": hosp.get("patient_id", ""),
                    "hospitalization_id": hosp["hospitalization_id"],
                    "order_dttm": order,
                    "collect_dttm": collect,
                    "result_dttm": result,
                    "fluid_category": fluid,
                    "fluid_name": self.name_from_category(fluid),
                    "method_category": "pcr",
                    "method_name": self.name_from_category("pcr"),
                    "organism_category": organism,
                    "organism_group": group,
                    "result_category": cat,
                    "result_name": self.name_from_category(cat),
                    "micro_order_name": test,
                    "reference_low": None,
                    "reference_high": None,
                    "result_units": None,
                    "lab_loinc_code": LOINC_CODES.get(test),
                })
        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            for col in ("order_dttm", "collect_dttm", "result_dttm"):
                df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def _tests(self, c, include_influenza):
        rng = self.rng
        out = []
        n = c.n_hours
        resp_ill = float(np.max(c.resp)) > 0.35 or (c.infected and c.infection_source == "pulmonary")
        if rng.random() < (0.9 if resp_ill else 0.35):
            h = float(rng.uniform(0, min(6, n * 0.3)))
            # A virus causes or accompanies some lung infections; a few others are incidental
            virus = None
            if c.infected and c.infection_source == "pulmonary" and rng.random() < 0.2:
                virus = str(rng.choice(["COVID-19 PCR", "Influenza A/B PCR", "RSV PCR"], p=[0.5, 0.3, 0.2]))
            elif rng.random() < 0.03:
                virus = "COVID-19 PCR"
            panel = ["COVID-19 PCR"]
            if resp_ill or rng.random() < 0.3:
                panel += ["Influenza A/B PCR", "RSV PCR"]
            if virus and virus not in panel:
                panel.append(virus)
            for test in panel:
                if test == "Influenza A/B PCR" and not include_influenza:
                    continue
                options = TESTS[test]
                pos = test == virus
                for k, (org, grp) in enumerate(options):
                    # Flu A/B is one specimen reported as two results; only one can be positive
                    org_pos = pos and (len(options) == 1 or k == int(rng.random() < 0.25))
                    out.append((test, h, org, grp, org_pos, "nasopharynx_upperairway"))

        # C. difficile after several days of antibiotics (or, rarely, without)
        abx_h = sum(a["end"] - a["start"] for a in c.antibiotics) if c.antibiotics else 0.0
        if n > 72:
            p_test = 0.25 if abx_h > 72 else 0.04
            if rng.random() < p_test:
                start = c.antibiotics[0]["start"] if c.antibiotics else 0.0
                h = float(min(start + rng.uniform(72, 168), n - 6))
                if h > 48:
                    pos = rng.random() < (0.15 if abx_h > 72 else 0.05)
                    out.append(("C. diff Toxin PCR", h, "clostridium_difficile", "clostridium_difficile",
                                bool(pos), "feces_stool"))
        return out
