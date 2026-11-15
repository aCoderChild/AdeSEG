from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_condition_engine_differs_only_in_explicit_record_and_prefix():
    s=(ROOT/'scripts/run_lean_indexed_spatial_rope.py').read_text()
    pairs=[('"""Evaluate position-preserving Lean rotary attention with archived prompts."""','"""Indexed Lean engine for causal first-read conditioning-memory experiments.\n\nThe recorded identity pins the original indexed baseline; engine_variant_sha256\nseparately identifies this runner, whose only change is the causal prefix check.\n"""'),
     ("'runner_sha256':digest(Path(__file__))","'runner_sha256':digest(ROOT/'scripts/run_lean_indexed_spatial_rope.py')"),
     ("'identity':identity,'reference_records_sha256':ref_done['records_sha256']", "'identity':identity,'engine_variant_sha256':digest(Path(__file__)),'engine_note':'identity pins original indexed baseline; this engine changes the causal prefix check','reference_records_sha256':ref_done['records_sha256']"),
     ("if a.sequence=='seq3' and masks[:3]!=ref['mask_sha256'][:3]: raise ValueError('Pre-intervention prefix changed')", "if a.sequence=='seq3' and masks[:2]!=ref['mask_sha256'][:2]: raise ValueError('Pre-intervention prefix changed')")]
    for before,after in pairs:
        assert s.count(before)==1
        s=s.replace(before,after)
    assert s==(ROOT/'scripts/run_lean_condition_indexed_engine.py').read_text()
