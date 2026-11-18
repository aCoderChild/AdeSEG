"""AdeSEG research APIs.

Two independent backbones:
  MedSAM2 Hiera-T/512 -> learned high-resolution skip adapter
  Lean-SAM2 Hiera-B+  -> learned multiscale channel-value refinement

Training-only objectives live in adeseg.losses, not the inference pipeline.
Historical source-bound code remains untouched in modeling/ for provenance.
"""
__version__ = "0.1.0"
