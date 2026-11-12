"""Generators for tables that exist only in CLIF 3.0.

These run only when the dataset is generated as CLIF 3.0 (the default).
They read the same per-stay course as the 2.1 generators
(``synthetic_clif.physiology``) and the 2.1 tables already generated, so the
new tables agree with the rest of the dataset. Their output already uses 3.0
vocabularies; ``synthetic_clif.clif3.convert`` only conforms columns and types.
"""

from synthetic_clif.generators.v3.clinical_notes import ClinicalNotesFactsGenerator
from synthetic_clif.generators.v3.dialysis import IntermittentDialysisGenerator
from synthetic_clif.generators.v3.drain import DrainGenerator, label_urine_output
from synthetic_clif.generators.v3.patient_attributes import PatientAttributesGenerator
from synthetic_clif.generators.v3.radiology import RadiologyGenerator
from synthetic_clif.generators.v3.scores import ScoresGenerator, model_registry
from synthetic_clif.generators.v3.validated_diagnosis import ValidatedDiagnosisGenerator

__all__ = [
    "ClinicalNotesFactsGenerator",
    "DrainGenerator",
    "IntermittentDialysisGenerator",
    "PatientAttributesGenerator",
    "RadiologyGenerator",
    "ScoresGenerator",
    "ValidatedDiagnosisGenerator",
    "label_urine_output",
    "model_registry",
]
