"""Scientific module boundaries for the proposed AdeSEG VPS method.

FGPI: initial target prompt generation (automatic, no GT).
RMFA: region-aware residual mask-logit refinement (learned).
TTMR: target-aware temporal token policy (currently a reference selector).

Low-level implementations remain at their original paths to keep historical
checkpoint source hashes and existing CLI imports stable.
"""
from adeseg.modules.fgpi import FGPI, FGPIConfig
from adeseg.modules.rmfa import RMFA, RMFAConfig
from adeseg.modules.ttmr import TTMR, TTMRConfig

__all__ = ["FGPI", "FGPIConfig", "RMFA", "RMFAConfig", "TTMR", "TTMRConfig"]
