#!/usr/bin/env python3
"""Describe score component scales in independently verified saved memory reads."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for name in ('capture','verification','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');v=read(a.verification/'verification.json')
    if not cap['passed'] or not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_priority_repeat.py') or cap['sequence']!=v['sequence'] or cap['targets']!=v['frames']:raise ValueError('Capture or diagnostic verification invalid')
    rows=[]
    for s in cap['snapshots']:
        path=a.capture/'source_snapshots'/f"{s['frame']:06d}.pt"
        if digest(path)!=s['sources_sha256']:raise ValueError('Source snapshot changed')
        snapshot=torch.load(path,map_location='cpu',weights_only=True);sources=snapshot['sources']
        if snapshot['frame']!=s['frame'] or not sources:raise ValueError('Source snapshot mismatch')
        arrays={k:torch.cat([x[k].float().reshape(-1) for x in sources]).numpy() for k in ('usage','contrast','boundary')}
        if len({len(x) for x in arrays.values()})!=1 or any(not np.isfinite(x).all() for x in arrays.values()):raise ValueError('Invalid source scores')
        stats={}
        for name,x in arrays.items():
            q=np.quantile(x,[.25,.5,.75]);stats[name]={'q25':float(q[0]),'median':float(q[1]),'q75':float(q[2]),'iqr':float(q[2]-q[0])}
        rows.append({'frame':s['frame'],'source_frames':[x['frame'] for x in sources],'tokens':len(arrays['usage']), 'statistics':stats,
          'weighted_usage_iqr':.5*stats['usage']['iqr'],'weighted_signed_contrast_iqr':.25*stats['contrast']['iqr'],'weighted_boundary_iqr':.25*stats['boundary']['iqr']})
    result={'scope':'Descriptive statistics of already retained older tokens, not causal attribution or a trained-attention evaluation.',
      'sequence':cap['sequence'],'frames':rows,'capture_sha256':digest(a.capture/'priority_capture_receipt.json'),'verification_sha256':digest(a.verification/'verification.json'),'source_sha256':digest(Path(__file__))}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Region score scales','','Interquartile ranges of the weighted score components across tokens previously retained by Lean. The boundary-prototype similarity has a different distribution than signed foreground/background contrast. These descriptive scales do not establish dominance of the combined ranking or causal usefulness.','','| Frame | Retained tokens | 0.5 × usage IQR | 0.25 × contrast IQR | 0.25 × boundary IQR |','| ---: | ---: | ---: | ---: | ---: |']
    for x in rows:lines.append(f"| {x['frame']} | {x['tokens']} | {x['weighted_usage_iqr']:.6f} | {x['weighted_signed_contrast_iqr']:.6f} | {x['weighted_boundary_iqr']:.6f} |")
    lines+=['','The signed contrast and boundary scores both use the initial automatic prompt; an inaccurate prompt can bias their interpretation.','']
    (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print(str(out/'report.md'))
if __name__=='__main__':main()
