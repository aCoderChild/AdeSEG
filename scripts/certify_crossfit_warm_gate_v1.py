#!/usr/bin/env python3
"""Certify the reconstructed warm gate once; children recheck bound inputs."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.crossfit_observer_capture_shared_v1 import BASE,DATA,EXISTING,phase_artifact,read,require
from scripts.run_crossfit_warm_gate_v1 import SOURCES as GATE_SOURCES,WARM_SHA,warm_plan,verify_old_mps_hooks

VIDEO_SOURCES=('scripts/certify_crossfit_warm_gate_v1.py','scripts/run_crossfit_warm_video_v1.py',
    'scripts/audit_crossfit_warm_video_v1.py','scripts/run_crossfit_warm_video_study_v1.py')

def sources():return {p:digest(ROOT/p) for p in (*GATE_SOURCES,*VIDEO_SOURCES)}


def reconstruct_gate(path):
    from scripts.audit_crossfit_warm_gate_v1 import audit
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to(BASE.resolve()) and path.name=='audit.json','Invalid warm gate')
    root=path.parent.parent;sha=phase_artifact(root,'audit',path);record=read(path)
    require(digest(path)==sha and record.get('passed') is True and record.get('warm_plan_sha256')==WARM_SHA and
            record.get('label_guard',{}).get('open_attempts')==0,'Warm gate identity differs')
    rebuilt=audit(root)
    require({k:v for k,v in record.items() if k!='label_guard'}==rebuilt and digest(path)==sha,
            'Warm gate reconstruction differs')
    require(record['seq3_shared_frames']==list(range(1,15)) and record['seq6_shared_frames']==[24,90],
            'Warm historical compatibility coverage differs')
    return record,sha


def verify_certificate(path):
    path=Path(path).resolve(strict=True);require(path.is_relative_to(BASE.resolve()) and path.name=='certificate.json',
                                                'Invalid warm certificate')
    done=read(path.parent/'completion.json');sha=digest(path);record=read(path)
    require(digest(path)==sha and done.get('passed') is True and done.get('certificate_sha256')==sha and
            done.get('label_guard',{}).get('open_attempts')==0 and record.get('passed') is True and
            record.get('warm_plan_sha256')==WARM_SHA and record.get('source_sha256')==sources() and
            record.get('certifier_sha256')==digest(Path(__file__)) and record.get('label_guard',{}).get('open_attempts')==0,
            'Warm certificate changed')
    warm=warm_plan();verify_old_mps_hooks(warm['old_identity'])
    gate=Path(record['warm_gate_path']).resolve(strict=True)
    require(gate.is_relative_to(BASE.resolve()) and phase_artifact(gate.parent.parent,'audit',gate)==record['warm_gate_sha256'],
            'Certified warm gate changed')
    gate_record=read(gate)
    require(digest(gate)==record['warm_gate_sha256'] and gate_record.get('passed') is True and
            gate_record.get('warm_plan_sha256')==WARM_SHA and gate_record.get('label_guard',{}).get('open_attempts')==0 and
            record.get('seq6_gate_capture')==gate_record.get('runs',{}).get('seq6_capture') and
            record.get('seq6_historical_reference')==warm['seq6_historical'],
            'Certified gate receipt differs')
    manifest=EXISTING/'manifest.json';require(digest(manifest)==record['existing_manifest_sha256']==warm['existing_manifest_sha256'],
                                             'Certified historical manifest changed')
    m=read(manifest);require(digest(manifest)==record['existing_manifest_sha256'],'Manifest changed while reading')
    checkpoint=Path(record['checkpoint_path']).resolve(strict=True)
    require(str(checkpoint)==str(Path(m['checkpoint_path']).resolve()) and
            digest(checkpoint)==record['checkpoint_sha256']==warm['old_identity']['checkpoint_sha256'],
            'Certified checkpoint changed')
    return record,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--warm-gate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Invalid certificate output')
    guard=install_mask_open_guard(DATA);warm=warm_plan();verify_old_mps_hooks(warm['old_identity'])
    gate,gate_sha=reconstruct_gate(a.warm_gate);manifest=EXISTING/'manifest.json'
    require(digest(manifest)==warm['existing_manifest_sha256'],'Historical manifest changed')
    m=read(manifest);checkpoint=Path(m['checkpoint_path']).resolve(strict=True)
    require(digest(manifest)==warm['existing_manifest_sha256'] and digest(checkpoint)==warm['old_identity']['checkpoint_sha256'] and
            guard['open_attempts']==0,'Certificate inputs differ')
    result={'passed':True,'scope':'Historical tensors and gate reconstructed at certification; old tensors must be rehashed when consumed.',
        'warm_plan_sha256':WARM_SHA,'warm_gate_path':str(a.warm_gate.resolve()),'warm_gate_sha256':gate_sha,
        'existing_manifest_sha256':digest(manifest),'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
        'seq6_gate_capture':gate['runs']['seq6_capture'],'seq6_historical_reference':warm['seq6_historical'],
        'source_sha256':sources(),'certifier_sha256':digest(Path(__file__)),'label_guard':dict(guard)}
    out.mkdir(parents=True,exist_ok=False);path=out/'certificate.json';path.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'certificate_sha256':digest(path),'label_guard':dict(guard)},indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'label_open_attempts':0}))

if __name__=='__main__':main()
