#!/usr/bin/env python3
"""Independently verify real raw-window feature-capture pipeline gates."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal
from modeling.lean_channel_feature_capture import sha


def read(p):return json.loads(p.read_text())
def check(ok,msg):
    if not ok:raise ValueError(msg)
def main():
    p=argparse.ArgumentParser()
    for k in ('first','second','raw-gate','feature-reference','feature-gate','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();fg=read(a.feature_gate/'verification.json');pre=read(a.preflight/'completion.json')
    check(fg['passed'] and fg['verifier_sha256']==digest(ROOT/'scripts/verify_lean_channel_feature_gates.py') and fg['preflight_sha256']==digest(a.preflight/'manifest.json') and pre['manifest_sha256']==fg['preflight_sha256'],'Prerequisites changed')
    reference=read(a.feature_reference/'feature_capture_receipt.json');outcomes=[];samples=[]
    for folder in (a.first,a.second):
        r=read(folder/'channel_feature_receipt.json');integrity=read(folder/'parameter_integrity_after.json')
        check(r['passed'] and r['mode']=='gate' and r['sequence']=='seq3' and r['targets']==fg['targets'],'Gate receipt invalid')
        check(r['runner_sha256']==digest(ROOT/'scripts/capture_lean_channel_features.py') and r['feature_gate_sha256']==digest(a.feature_gate/'verification.json') and r['preflight_sha256']==pre['manifest_sha256'],'Source changed')
        check(r['records_sha256']==digest(folder/'records.json') and r['events_sha256']==digest(folder/'token_window_events.json'),'Trace changed')
        check(integrity['unchanged'] and not integrity['changed_names'] and integrity['before']==integrity['after'],'Model changed')
        for name in ('records.json','positions.json','selections.json','memory_events.json','token_window_events.json'):check(digest(folder/name)==digest(a.raw_gate/name),'Raw baseline changed')
        record=read(folder/'records.json');check(record['mask_sha256']==[digest(folder/f'{i:06d}.png') for i in range(record['frames'])],'Saved mask changed')
        check(r['feature_pipeline_sha256']==hashlib.sha256(json.dumps(r['pipeline'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'Pipeline digest differs')
        values=[]
        for item in r['features']:
            frame=item['frame'];path=folder/'features'/f'{frame:06d}.pt';check(item['file_sha256']==digest(path),'Feature file changed')
            data=torch.load(path,map_location='cpu',weights_only=True);ref=next(x for x in reference['features'] if x['frame']==frame)
            for key in ('feature','foreground','background'):check(item[key+'_sha256']==sha(data[key])==ref[key+'_sha256'],'Feature mismatch')
            values.append(data)
        check({x['frame'] for x in r['features']}==set(fg['targets']),'Coverage mismatch')
        outcomes.append(r);samples.append(values)
    check(outcomes[0]==outcomes[1] and equal(samples[0],samples[1]),'Independent gates differ')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    res={'passed':True,'feature_pipeline_sha256':outcomes[0]['feature_pipeline_sha256'],'preflight_sha256':pre['manifest_sha256'],
      'runner_sha256':digest(ROOT/'scripts/capture_lean_channel_features.py'),'verifier_sha256':digest(Path(__file__)),
      'scope':'Two no-op engineering gates; no trained model or efficacy claim.'}
    (out/'verification.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()
