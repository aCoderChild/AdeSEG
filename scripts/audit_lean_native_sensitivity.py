#!/usr/bin/env python3
"""Recompute and check the saved leave-one-sequence-out analysis."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for name in ('comparison-dir','analysis','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();comparison=read(a.comparison_dir/'comparison.json');historical=read(a.comparison_dir/'independent_mask_audit.json');res=read(a.analysis/'results.json');done=read(a.analysis/'completion.json')
    if not done['passed'] or digest(a.analysis/'results.json')!=done['results_sha256'] or digest(a.analysis/'report.md')!=done['report_sha256'] or res['source_sha256']!=digest(ROOT/'scripts/analyze_lean_native_sensitivity.py') or res['comparison_sha256']!=digest(a.comparison_dir/'comparison.json') or not historical['passed'] or historical['audit_source_sha256']!=digest(ROOT/'scripts/audit_lean_polypgen_results.py') or historical['comparison_sha256']!=res['comparison_sha256']:
        raise ValueError('Source or historical audit mismatch')
    rows=comparison['sequences'];d={x['sequence']:x['lean_dice']-x['native_dice'] for x in rows}
    if len(d)!=21 or set(d)!={x['sequence'] for x in res['leave_one_out']}:raise ValueError('Sequence set mismatch')
    mean=sum(d.values())/len(d)
    if abs(mean-res['overall_mean_delta'])>1e-12 or [sum(x>0 for x in d.values()),sum(x<0 for x in d.values()),sum(x==0 for x in d.values())]!=[res['lean_wins'],res['native_wins'],res['ties']]:raise ValueError('Aggregate differs')
    for row in res['leave_one_out']:
        seq=row['sequence'];other=sum(v for k,v in d.items() if k!=seq)/(len(d)-1)
        if abs(row['delta']-d[seq])>1e-12 or abs(row['mean_without']-other)>1e-12:raise ValueError('Leave-one-out differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    receipt={'passed':True,'sequences':len(d),'analysis_results_sha256':digest(a.analysis/'results.json'),'analysis_report_sha256':digest(a.analysis/'report.md'),'source_sha256':digest(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
