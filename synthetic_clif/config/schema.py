"""CLIF 2.1.0 table schema definitions (from official MySQL DDL)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""

    name: str
    dtype: str  # pandas dtype string
    nullable: bool = True
    mcide_category: Optional[str] = None  # Reference to mCIDE category if applicable
    description: str = ""


@dataclass
class TableSchema:
    """Schema definition for a CLIF table."""

    name: str
    columns: list[ColumnSchema]
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: dict[str, str] = field(default_factory=dict)  # column -> referenced_table.column
    is_beta: bool = True  # True for beta tables, False for concept tables

    def column_names(self) -> list[str]:
        """Return list of column names."""
        return [col.name for col in self.columns]

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Get column schema by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None


class CLIFSchema:
    """CLIF 2.1.0 schema definitions for all tables."""

    # Beta Tables (16)

    PATIENT = TableSchema(
        name="patient",
        columns=[
            ColumnSchema("patient_id", "string", nullable=False),
            ColumnSchema("race_name", "string"),
            ColumnSchema("race_category", "string", mcide_category="race"),
            ColumnSchema("ethnicity_name", "string"),
            ColumnSchema("ethnicity_category", "string", mcide_category="ethnicity"),
            ColumnSchema("sex_name", "string"),
            ColumnSchema("sex_category", "string", mcide_category="sex"),
            ColumnSchema("birth_date", "datetime64[ns]"),
            ColumnSchema("death_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("language_name", "string"),
            ColumnSchema("language_category", "string", mcide_category="language"),
        ],
        primary_key=["patient_id"],
        is_beta=True,
    )

    HOSPITALIZATION = TableSchema(
        name="hospitalization",
        columns=[
            ColumnSchema("patient_id", "string", nullable=False),
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("hospitalization_joined_id", "string"),
            ColumnSchema("admission_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("discharge_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("age_at_admission", "int64"),
            ColumnSchema("admission_type_name", "string"),
            ColumnSchema("admission_type_category", "string", mcide_category="admission_type"),
            ColumnSchema("discharge_name", "string"),
            ColumnSchema("discharge_category", "string", mcide_category="discharge"),
            ColumnSchema("zipcode_nine_digit", "string"),
            ColumnSchema("zipcode_five_digit", "string"),
            ColumnSchema("census_block_code", "string"),
            ColumnSchema("census_block_group_code", "string"),
            ColumnSchema("census_tract", "string"),
            ColumnSchema("state_code", "string"),
            ColumnSchema("county_code", "string"),
            ColumnSchema("fips_version", "string"),
        ],
        primary_key=["hospitalization_id"],
        foreign_keys={"patient_id": "patient.patient_id"},
        is_beta=True,
    )

    ADT = TableSchema(
        name="adt",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("hospital_id", "string"),
            ColumnSchema("hospital_type", "string"),
            ColumnSchema("in_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("out_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("location_name", "string"),
            ColumnSchema("location_category", "string", mcide_category="location"),
            ColumnSchema("location_type", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    VITALS = TableSchema(
        name="vitals",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("vital_name", "string"),
            ColumnSchema("vital_category", "string", nullable=False, mcide_category="vital"),
            ColumnSchema("vital_value", "float64"),
            ColumnSchema("meas_site_name", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    LABS = TableSchema(
        name="labs",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("lab_order_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("lab_collect_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("lab_result_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("lab_order_name", "string"),
            ColumnSchema("lab_order_category", "string", mcide_category="lab_order"),
            ColumnSchema("lab_name", "string"),
            ColumnSchema("lab_category", "string", nullable=False, mcide_category="lab"),
            ColumnSchema("lab_value", "string"),
            ColumnSchema("lab_value_numeric", "float64"),
            ColumnSchema("reference_unit", "string"),
            ColumnSchema("lab_specimen_name", "string"),
            ColumnSchema("lab_specimen_category", "string", mcide_category="lab_specimen"),
            ColumnSchema("lab_loinc_code", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    RESPIRATORY_SUPPORT = TableSchema(
        name="respiratory_support",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("device_name", "string"),
            ColumnSchema("device_id", "string"),
            ColumnSchema("device_category", "string", nullable=False, mcide_category="respiratory_device"),
            ColumnSchema("vent_brand_name", "string"),
            ColumnSchema("mode_name", "string"),
            ColumnSchema("mode_category", "string", mcide_category="respiratory_mode"),
            ColumnSchema("tracheostomy", "int64"),
            ColumnSchema("fio2_set", "float64"),
            ColumnSchema("lpm_set", "float64"),
            ColumnSchema("tidal_volume_set", "float64"),
            ColumnSchema("resp_rate_set", "float64"),
            ColumnSchema("pressure_control_set", "float64"),
            ColumnSchema("pressure_support_set", "float64"),
            ColumnSchema("flow_rate_set", "float64"),
            ColumnSchema("peak_inspiratory_pressure_set", "float64"),
            ColumnSchema("inspiratory_time_set", "float64"),
            ColumnSchema("peep_set", "float64"),
            ColumnSchema("tidal_volume_obs", "float64"),
            ColumnSchema("resp_rate_obs", "float64"),
            ColumnSchema("plateau_pressure_obs", "float64"),
            ColumnSchema("peak_inspiratory_pressure_obs", "float64"),
            ColumnSchema("peep_obs", "float64"),
            ColumnSchema("minute_vent_obs", "float64"),
            ColumnSchema("mean_airway_pressure_obs", "float64"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    MEDICATION_ADMIN_CONTINUOUS = TableSchema(
        name="medication_admin_continuous",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("med_order_id", "string"),
            ColumnSchema("admin_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("med_name", "string"),
            ColumnSchema("med_category", "string", nullable=False, mcide_category="medication"),
            ColumnSchema("med_group", "string"),
            ColumnSchema("med_route_name", "string"),
            ColumnSchema("med_route_category", "string", mcide_category="med_route"),
            ColumnSchema("med_dose", "float64"),
            ColumnSchema("med_dose_unit", "string"),
            ColumnSchema("mar_action_name", "string"),
            ColumnSchema("mar_action_category", "string", mcide_category="mar_action"),
            ColumnSchema("mar_action_group", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    MEDICATION_ADMIN_INTERMITTENT = TableSchema(
        name="medication_admin_intermittent",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("med_order_id", "string"),
            ColumnSchema("admin_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("med_name", "string"),
            ColumnSchema("med_category", "string", nullable=False, mcide_category="medication"),
            ColumnSchema("med_group", "string"),
            ColumnSchema("med_route_name", "string"),
            ColumnSchema("med_route_category", "string", mcide_category="med_route"),
            ColumnSchema("med_dose", "float64"),
            ColumnSchema("med_dose_unit", "string"),
            ColumnSchema("mar_action_name", "string"),
            ColumnSchema("mar_action_category", "string", mcide_category="mar_action"),
            ColumnSchema("mar_action_group", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    MICROBIOLOGY_CULTURE = TableSchema(
        name="microbiology_culture",
        columns=[
            ColumnSchema("patient_id", "string", nullable=False),
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("organism_id", "string"),
            ColumnSchema("order_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("collect_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("result_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("fluid_name", "string"),
            ColumnSchema("fluid_category", "string", mcide_category="culture_fluid"),
            ColumnSchema("method_name", "string"),
            ColumnSchema("method_category", "string", mcide_category="culture_method"),
            ColumnSchema("organism_name", "string"),
            ColumnSchema("organism_category", "string", mcide_category="organism"),
            ColumnSchema("organism_group", "string", mcide_category="organism_group"),
            ColumnSchema("lab_loinc_code", "string"),
        ],
        foreign_keys={
            "patient_id": "patient.patient_id",
            "hospitalization_id": "hospitalization.hospitalization_id",
        },
        is_beta=True,
    )

    MICROBIOLOGY_SUSCEPTIBILITY = TableSchema(
        name="microbiology_susceptibility",
        columns=[
            ColumnSchema("organism_id", "string", nullable=False),
            ColumnSchema("antimicrobial_name", "string"),
            ColumnSchema("antimicrobial_category", "string", mcide_category="antimicrobial"),
            ColumnSchema("sensitivity_name", "string"),
            ColumnSchema("susceptibility_name", "string"),
            ColumnSchema("susceptibility_category", "string", mcide_category="susceptibility"),
        ],
        foreign_keys={"organism_id": "microbiology_culture.organism_id"},
        is_beta=True,
    )

    PATIENT_ASSESSMENTS = TableSchema(
        name="patient_assessments",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("assessment_name", "string"),
            ColumnSchema("assessment_category", "string", nullable=False, mcide_category="assessment"),
            ColumnSchema("assessment_group", "string"),
            ColumnSchema("numerical_value", "float64"),
            ColumnSchema("categorical_value", "string"),
            ColumnSchema("text_value", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    PATIENT_PROCEDURES = TableSchema(
        name="patient_procedures",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("billing_provider_id", "string"),
            ColumnSchema("performing_provider_id", "string"),
            ColumnSchema("procedure_code", "string"),
            ColumnSchema("procedure_code_format", "string"),
            ColumnSchema("procedure_billed_dttm", "datetime64[ns, UTC]"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    HOSPITAL_DIAGNOSIS = TableSchema(
        name="hospital_diagnosis",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("diagnosis_code", "string"),
            ColumnSchema("diagnosis_code_format", "string"),
            ColumnSchema("diagnosis_primary", "int64"),
            ColumnSchema("poa_present", "int64"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    CRRT_THERAPY = TableSchema(
        name="crrt_therapy",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("device_id", "string"),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("crrt_mode_name", "string"),
            ColumnSchema("crrt_mode_category", "string", mcide_category="crrt_mode"),
            ColumnSchema("dialysis_machine_name", "string"),
            ColumnSchema("blood_flow_rate", "float64"),
            ColumnSchema("pre_filter_replacement_fluid_rate", "float64"),
            ColumnSchema("post_filter_replacement_fluid_rate", "float64"),
            ColumnSchema("dialysate_flow_rate", "float64"),
            ColumnSchema("ultrafiltration_out", "float64"),
        ],
        is_beta=True,
    )

    POSITION = TableSchema(
        name="position",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("position_name", "string"),
            ColumnSchema("position_category", "string", mcide_category="position"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=True,
    )

    CODE_STATUS = TableSchema(
        name="code_status",
        columns=[
            ColumnSchema("patient_id", "string", nullable=False),
            ColumnSchema("start_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("code_status_name", "string"),
            ColumnSchema("code_status_category", "string", mcide_category="code_status"),
        ],
        foreign_keys={"patient_id": "patient.patient_id"},
        is_beta=True,
    )

    # Concept Tables (12)

    CLINICAL_TRIAL = TableSchema(
        name="clinical_trial",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("trial_id", "string"),
            ColumnSchema("trial_name", "string"),
            ColumnSchema("arm_id", "string"),
            ColumnSchema("consent_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("randomized_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("withdrawal_dttm", "datetime64[ns, UTC]"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    ECMO_MCS = TableSchema(
        name="ecmo_mcs",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("device_name", "string"),
            ColumnSchema("device_category", "string", mcide_category="ecmo_device"),
            ColumnSchema("mcs_group", "string"),
            ColumnSchema("device_metric_name", "string"),
            ColumnSchema("device_rate", "float64"),
            ColumnSchema("sweep", "float64"),
            ColumnSchema("flow", "float64"),
            ColumnSchema("fdO2", "float64"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    INTAKE_OUTPUT = TableSchema(
        name="intake_output",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("intake_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("fluid_name", "string"),
            ColumnSchema("amount", "float64"),
            ColumnSchema("in_out_flag", "int64"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    INVASIVE_HEMODYNAMICS = TableSchema(
        name="invasive_hemodynamics",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("recorded_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("measure_name", "string"),
            ColumnSchema("measure_category", "string", mcide_category="hemodynamic"),
            ColumnSchema("measure_value", "float64"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    KEY_ICU_ORDERS = TableSchema(
        name="key_icu_orders",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("order_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("order_name", "string"),
            ColumnSchema("order_category", "string", mcide_category="icu_order"),
            ColumnSchema("order_status_name", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    MEDICATION_ORDERS = TableSchema(
        name="medication_orders",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("med_order_id", "string", nullable=False),
            ColumnSchema("order_start_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("order_end_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("ordered_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("med_name", "string"),
            ColumnSchema("med_category", "string", mcide_category="medication"),
            ColumnSchema("med_group", "string"),
            ColumnSchema("med_order_status_name", "string"),
            ColumnSchema("med_order_status_category", "string", mcide_category="med_order_status"),
            ColumnSchema("med_route_name", "string"),
            ColumnSchema("med_dose", "float64"),
            ColumnSchema("med_dose_unit", "string"),
            ColumnSchema("med_frequency", "string"),
            ColumnSchema("prn", "boolean"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    MICROBIOLOGY_NONCULTURE = TableSchema(
        name="microbiology_nonculture",
        columns=[
            ColumnSchema("patient_id", "string", nullable=False),
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("result_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("collect_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("order_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("fluid_name", "string"),
            ColumnSchema("fluid_category", "string", mcide_category="culture_fluid"),
            ColumnSchema("method_name", "string"),
            ColumnSchema("method_category", "string", mcide_category="culture_method"),
            ColumnSchema("micro_order_name", "string"),
            ColumnSchema("organism_category", "string", mcide_category="organism"),
            ColumnSchema("organism_group", "string", mcide_category="organism_group"),
            ColumnSchema("result_name", "string"),
            ColumnSchema("result_category", "string", mcide_category="micro_result"),
            ColumnSchema("reference_low", "float64"),
            ColumnSchema("reference_high", "float64"),
            ColumnSchema("result_units", "string"),
            ColumnSchema("lab_loinc_code", "string"),
        ],
        foreign_keys={
            "patient_id": "patient.patient_id",
            "hospitalization_id": "hospitalization.hospitalization_id",
        },
        is_beta=False,
    )

    PATIENT_DIAGNOSIS = TableSchema(
        name="patient_diagnosis",
        columns=[
            ColumnSchema("patient_id", "string", nullable=False),
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("diagnosis_code", "string"),
            ColumnSchema("diagnosis_code_format", "string"),
            ColumnSchema("source_type", "string"),
            ColumnSchema("start_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("end_dttm", "datetime64[ns, UTC]"),
        ],
        foreign_keys={
            "patient_id": "patient.patient_id",
            "hospitalization_id": "hospitalization.hospitalization_id",
        },
        is_beta=False,
    )

    PLACE_BASED_INDEX = TableSchema(
        name="place_based_index",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("index_name", "string"),
            ColumnSchema("index_value", "float64"),
            ColumnSchema("index_version", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    PROVIDER = TableSchema(
        name="provider",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("provider_id", "string"),
            ColumnSchema("start_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("stop_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("provider_role_name", "string"),
            ColumnSchema("provider_role_category", "string", mcide_category="provider_role"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    THERAPY_DETAILS = TableSchema(
        name="therapy_details",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("session_start_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("therapy_element_name", "string"),
            ColumnSchema("therapy_element_category", "string", mcide_category="therapy"),
            ColumnSchema("therapy_element_value", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    TRANSFUSION = TableSchema(
        name="transfusion",
        columns=[
            ColumnSchema("hospitalization_id", "string", nullable=False),
            ColumnSchema("transfusion_start_dttm", "datetime64[ns, UTC]", nullable=False),
            ColumnSchema("transfusion_end_dttm", "datetime64[ns, UTC]"),
            ColumnSchema("component_name", "string"),
            ColumnSchema("attribute_name", "string"),
            ColumnSchema("volume_transfused", "float64"),
            ColumnSchema("volume_units", "string"),
            ColumnSchema("product_code", "string"),
        ],
        foreign_keys={"hospitalization_id": "hospitalization.hospitalization_id"},
        is_beta=False,
    )

    @classmethod
    def get_all_schemas(cls) -> dict[str, TableSchema]:
        """Return all table schemas."""
        return {
            "patient": cls.PATIENT,
            "hospitalization": cls.HOSPITALIZATION,
            "adt": cls.ADT,
            "vitals": cls.VITALS,
            "labs": cls.LABS,
            "respiratory_support": cls.RESPIRATORY_SUPPORT,
            "medication_admin_continuous": cls.MEDICATION_ADMIN_CONTINUOUS,
            "medication_admin_intermittent": cls.MEDICATION_ADMIN_INTERMITTENT,
            "microbiology_culture": cls.MICROBIOLOGY_CULTURE,
            "microbiology_susceptibility": cls.MICROBIOLOGY_SUSCEPTIBILITY,
            "patient_assessments": cls.PATIENT_ASSESSMENTS,
            "patient_procedures": cls.PATIENT_PROCEDURES,
            "hospital_diagnosis": cls.HOSPITAL_DIAGNOSIS,
            "crrt_therapy": cls.CRRT_THERAPY,
            "position": cls.POSITION,
            "code_status": cls.CODE_STATUS,
            "clinical_trial": cls.CLINICAL_TRIAL,
            "ecmo_mcs": cls.ECMO_MCS,
            "intake_output": cls.INTAKE_OUTPUT,
            "invasive_hemodynamics": cls.INVASIVE_HEMODYNAMICS,
            "key_icu_orders": cls.KEY_ICU_ORDERS,
            "medication_orders": cls.MEDICATION_ORDERS,
            "microbiology_nonculture": cls.MICROBIOLOGY_NONCULTURE,
            "patient_diagnosis": cls.PATIENT_DIAGNOSIS,
            "place_based_index": cls.PLACE_BASED_INDEX,
            "provider": cls.PROVIDER,
            "therapy_details": cls.THERAPY_DETAILS,
            "transfusion": cls.TRANSFUSION,
        }

    @classmethod
    def get_beta_schemas(cls) -> dict[str, TableSchema]:
        """Return only beta table schemas."""
        return {k: v for k, v in cls.get_all_schemas().items() if v.is_beta}

    @classmethod
    def get_concept_schemas(cls) -> dict[str, TableSchema]:
        """Return only concept table schemas."""
        return {k: v for k, v in cls.get_all_schemas().items() if not v.is_beta}
