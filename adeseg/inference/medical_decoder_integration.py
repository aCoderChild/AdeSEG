"""Scoped injection of the medical adapter into native MedSAM2 decoder skips."""
from modeling.medical_highres.integration import attach_medical_highres_adapter

__all__ = ["attach_medical_highres_adapter"]
