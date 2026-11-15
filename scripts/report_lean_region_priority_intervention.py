#!/usr/bin/env python3
"""Summarize independently audited, repeated fixed-read deletion results."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for key in ('audit','replay','verification','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();v=read(a.verification/'verification.json');audit=read(a.audit/'results.json');replay=read(a.replay/'completion.json')
    if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_priority_repeat.py') or v['audit_sha256']!=digest(a.audit/'results.json') or v['first_completion_sha256']!=digest(a.replay/'completion.json') or not replay['passed']:
        raise ValueError('Audited repeat verification changed')
    byframe={e['frame']:e for e in replay['events']};methods=['usage','contrast','distributed','three_region','foreground','shuffled','rgb_contrast']
    lines=[f"# {audit['sequence']}: matched fixed-read token deletions",'',
      'Exploratory, retrospective same-state test. Each method deletes 1,024 older memory tokens; no deletion changes the later video state. Full conditioning, newest memory and pointer tokens are preserved. Positive Dice difference means deleting those tokens improved the fixed read; negative means deletion hurt it. This is not a full-video score or an independent generalization estimate.','',
      '| Frame | Label present | No-op Dice | '+ ' | '.join(methods)+' |','| ---: | :---: | ---: | '+' | '.join(['---:']*len(methods))+' |']
    for row in audit['frames']:
        base=row['scores']['noop']['dice'];vals=[row['scores'][k]['dice']-base for k in methods]
        lines.append(f"| {row['frame']} | {'yes' if row['present'] else 'no'} | {base:.6f} | "+' | '.join(f'{x:+.6f}' for x in vals)+' |')
    lines+=['','For labeled-empty frames, see the independently scored false-alarm fields in `results.json`; Dice alone is not sufficient.','',
      '| Frame | Three-region ∩ distributed | Three-region ∩ shuffled | Three-region ∩ RGB | Distinct prediction masks |','| ---: | ---: | ---: | ---: | ---: |']
    for row in audit['frames']:
        e=byframe[row['frame']];chosen=set(e['deleted']['three_region'])
        nums=[len(chosen&set(e['deleted'][k])) for k in ('distributed','shuffled','rgb_contrast')]
        lines.append(f"| {row['frame']} | {nums[0]} / 1024 | {nums[1]} / 1024 | {nums[2]} / 1024 | {len(set(e['masks'].values()))} / 8 |")
    lines+=['','Similarity in deletion sets is descriptive. Deletion effects are conditional on the remaining memory and attention normalization. Prompt-derived prototypes may be contaminated by initial segmentation errors; the boundary descriptor may represent illumination or surrounding tissue.','']
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'report_sha256':digest(out/'report.md'),'verification_sha256':digest(a.verification/'verification.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n')
    print(str(out/'report.md'))
if __name__=='__main__':main()
