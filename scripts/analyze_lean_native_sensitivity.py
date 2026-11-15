#!/usr/bin/env python3
"""Descriptive leave-one-sequence-out check on an independently audited comparison."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    path=a.comparison_dir/'comparison.json';c=json.loads(path.read_text());audit=json.loads((a.comparison_dir/'independent_mask_audit.json').read_text())
    if not audit['passed'] or audit['audit_source_sha256']!=digest(ROOT/'scripts/audit_lean_polypgen_results.py') or audit['comparison_sha256']!=digest(path) or audit['targets']!=c['target_frames'] or audit['saved_masks_scored']!=2*c['target_frames']:raise ValueError('Comparison audit changed')
    rows=c['sequences'];d=np.asarray([r['lean_dice']-r['native_dice'] for r in rows],dtype=np.float64)
    if len(rows)!=21 or not np.isfinite(d).all() or not np.isclose(d.mean(),c['mean_delta'],atol=1e-12):raise ValueError('Unexpected aggregate')
    leave=[{'sequence':r['sequence'],'delta':float(di),'mean_without':float((d.sum()-di)/(len(d)-1))} for r,di in zip(rows,d)]
    results={'scope':'Descriptive sensitivity analysis of an already-inspected GT-initialized comparison; not a new benchmark or causal analysis.',
      'sequence_count':len(rows),'overall_mean_delta':float(d.mean()),'lean_wins':int((d>0).sum()),'native_wins':int((d<0).sum()),'ties':int((d==0).sum()),
      'leave_one_out':leave,'comparison_sha256':digest(path),'independent_audit_sha256':digest(a.comparison_dir/'independent_mask_audit.json'),'source_sha256':digest(Path(__file__))}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Lean versus native: sequence sensitivity','',results['scope'],'',f"Overall Lean-minus-native Dice: {d.mean():+.6f}. Lean higher / native higher / ties: {results['lean_wins']} / {results['native_wins']} / {results['ties']}.",'',
      '| Omitted sequence | That sequence’s delta | Mean delta without it |','| --- | ---: | ---: |']
    for x in leave:lines.append(f"| {x['sequence']} | {x['delta']:+.6f} | {x['mean_without']:+.6f} |")
    lines+=['','Both methods used matched GT-mask initialization in the original comparison, and the original Lean CPU implementation had a later-identified rotary-frequency discrepancy. Removing a sequence post hoc does not establish generalization or causality.','']
    (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print(json.dumps({'passed':True,'overall_delta':float(d.mean()),'wins':[results['lean_wins'],results['native_wins'],results['ties']]},indent=2))
if __name__=='__main__':main()
