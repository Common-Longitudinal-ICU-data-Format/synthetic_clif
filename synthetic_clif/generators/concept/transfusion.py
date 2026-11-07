"""Transfusion generator.

Blood products follow the hospital course: RBCs for a low haemoglobin,
platelets for thrombocytopenia (or bleeding), plasma and cryoprecipitate for
bleeding or a high INR. Each unit is one row, hung back to back, with the
volumes that the intake/output table also records.
"""

from datetime import timedelta

import pandas as pd

from synthetic_clif.generators.base import BaseGenerator
from synthetic_clif.physiology.course import course_for_row
from synthetic_clif.physiology.events import transfusion_units


class TransfusionGenerator(BaseGenerator):
    """Generate synthetic transfusion data from the shared course."""

    PRODUCT_CODES = {
        "Packed RBCs": "E0027",
        "Fresh Frozen Plasma": "E0032",
        "Platelets": "E0033",
        "Cryoprecipitate": "E0034",
    }

    COLUMNS = ["hospitalization_id", "transfusion_start_dttm", "transfusion_end_dttm",
               "component_name", "attribute_name", "volume_transfused", "volume_units", "product_code"]

    def generate(self, hospitalizations_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, hosp in hospitalizations_df.iterrows():
            course = course_for_row(hosp)
            if course is None or not course.transfusions:
                continue
            admit = hosp["admission_dttm"]
            discharge = hosp["discharge_dttm"]
            if pd.isna(discharge):
                discharge = admit + timedelta(days=5)
            for u in transfusion_units(course):
                start = admit + timedelta(hours=u["start_h"])
                end = admit + timedelta(hours=u["end_h"])
                if end > discharge:
                    continue
                records.append({
                    "hospitalization_id": hosp["hospitalization_id"],
                    "transfusion_start_dttm": start,
                    "transfusion_end_dttm": end,
                    "component_name": u["component"],
                    "attribute_name": u["attribute"],
                    "volume_transfused": u["volume"],
                    "volume_units": "mL",
                    "product_code": self.PRODUCT_CODES.get(u["component"]),
                })

        df = pd.DataFrame(records, columns=self.COLUMNS)
        if len(df) > 0:
            df["transfusion_start_dttm"] = pd.to_datetime(df["transfusion_start_dttm"], utc=True)
            df["transfusion_end_dttm"] = pd.to_datetime(df["transfusion_end_dttm"], utc=True)
        return df
