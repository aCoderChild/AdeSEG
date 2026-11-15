#!/usr/bin/env python3
"""Independently check repeated author-gate feature interface receipts."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from modeling.lean_channel_attention import author_sha256


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for key in ('first','second','cropped-reference','prompt-gate-a','prompt-gate-b','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();records=[]
    ga=read(a.prompt_gate_a/'gate.json');gb=read(a.prompt_gate_b/'gate.json')
    if ga!=gb or ga['setup']!=read(a.cropped_reference/'setup.json') or digest(a.prompt_gate_a/'prompt.png')!=ga['files']['prompt.png'] or digest(a.prompt_gate_b/'prompt.png')!=gb['files']['prompt.png'] or digest(a.cropped_reference/'prompt.png')!=ga['files']['prompt.png']:
        raise ValueError('Archived prompt image is not independently pinned')
    for folder in (a.first,a.second):
        r=read(folder/'interface_verification.json');integrity=read(folder/'parameter_integrity_after.json')
        if not r['passed'] or r['scope']!='zero-weight engineering gate, no learned inference or video propagation' or r['sequence']!='seq23' or r['frames']!=[0,8] or r['feature_shape']!=[32,256,256] or r['max_abs_baseline_error']>1e-6:raise ValueError('Interface check invalid')
        if r['author_cbam_sha256']!=author_sha256() or r['adapter_sha256']!=digest(ROOT/'modeling/lean_channel_attention.py') or r['runner_sha256']!=digest(ROOT/'scripts/verify_lean_channel_attention_interface.py'):raise ValueError('Source changed')
        if r['prompt_sha256']!=ga['files']['prompt.png']:raise ValueError('Prompt identity differs')
        if not integrity['unchanged'] or integrity['before']!=integrity['after'] or integrity['changed_names']:raise ValueError('Predictor weights changed')
        records.append(r)
    if records[0]!=records[1]:raise ValueError('Independent feature checks differ')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'source_reports_sha256':[digest(x/'interface_verification.json') for x in (a.first,a.second)],
      'max_abs_baseline_error':records[0]['max_abs_baseline_error'],'feature_shape':records[0]['feature_shape'],'adapter_sha256':records[0]['adapter_sha256'],
      'verifier_sha256':digest(Path(__file__)),'scope':'Fixed uniform-weight interface check only; no learned inference or segmentation efficacy.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
