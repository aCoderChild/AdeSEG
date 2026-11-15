#!/usr/bin/env python3
"""Independently validate Lean CPU RoPE correction engineering receipts."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from modeling.lean_rope_cpu_repeat_fix import corrected_cpu_key_frequency_repeat  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_lean_rope_cpu_fix import checked_gate  # noqa: E402


def read(path): return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replay-a','replay-b','fixed-a','fixed-b','merged','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    identity=read(a.replay_a/'setup.json')['identity']
    replay=[checked_gate(path,identity,'replay') for path in (a.replay_a,a.replay_b)]
    fixed=[checked_gate(path,identity,'fixed_gate') for path in (a.fixed_a,a.fixed_b)]
    if replay[0] != replay[1] or fixed[0] != fixed[1]: raise ValueError('Independent gate records differ')
    merged=read(a.merged/'completion.json')
    if not merged['passed'] or digest(a.merged/'results.json') != merged['results_sha256'] or identity['merged_results_sha256'] != merged['results_sha256']:
        raise ValueError('Archived reference changed')
    location=Path(read(a.merged/'locations.json')['seq3']); old=read(location/'seq3'/'frames.json')
    if replay[0]['mask_sha256'] != [x['mask_sha256'] for x in old]: raise ValueError('No-op differs from archive')
    first=fixed[0]['first_affected_frame']
    if first != replay[0]['first_affected_frame'] or any(fixed[0]['mask_sha256'][i] != old[i]['mask_sha256'] for i in range(first)):
        raise ValueError('Pre-fix prefix differs')
    trace=read(a.fixed_a/'frequency_trace.json')
    if not trace or min(x['frame'] for x in trace) != first or any(x['repeat'] <= 1 for x in trace):
        raise ValueError('Frequency trace malformed')
    selections=read(a.fixed_a/'selections.json')
    if not selections or any(x['candidates'] != 4096 or x['kept'] != 1024 for x in selections):
        raise ValueError('Original 25% token budget differs')
    # Confirm the scoped implementation reproduces the source CUDA expression
    # without needing a CUDA device; this is not whole-model GPU parity.
    from types import SimpleNamespace
    def released_cpu(base,count):
        return base if count == 1 else base.unsqueeze(1).expand(-1,count,-1).reshape(-1,base.shape[-1])
    module=SimpleNamespace(_repeat_freqs_for_k=released_cpu)
    base=torch.tensor([[1+1j],[2+2j],[3+3j]])
    if torch.equal(released_cpu(base,2),base.repeat(2,1)):
        raise ValueError('Fixture fails to expose the discrepancy')
    with corrected_cpu_key_frequency_repeat(module):
        if not torch.equal(module._repeat_freqs_for_k(base,2),base.repeat(2,1)):
            raise ValueError('Corrected fixture differs from CUDA expression')
    if module._repeat_freqs_for_k is not released_cpu: raise ValueError('Helper not restored')
    out=validate_output(a.output)
    if out.exists(): raise FileExistsError(out)
    result={'passed':True,'scope':'Engineering only; CPU helper equals the CUDA-branch expression, not a whole-model CPU/GPU equivalence claim.',
            'identity':identity,'replay_records_sha256':digest(a.replay_a/'records.json'),
            'fixed_records_sha256':digest(a.fixed_a/'records.json'), 'first_affected_frame':first,
            'prefix_frames_identical':first,'affected_calls':len(trace),
            'changed_frame_masks':sum(x != y for x,y in zip(fixed[0]['mask_sha256'],replay[0]['mask_sha256'])),
            'frames':len(old),'selection_events':len(selections),'spatial_tokens_kept':1024,'spatial_candidates':4096,
            'source_sha256':digest(Path(__file__))}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='identity'},indent=2))


if __name__=='__main__':main()
