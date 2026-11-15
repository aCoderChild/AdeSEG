#!/usr/bin/env python3
"""Draft separate 143-read handoff binding bounded seq11 and seq6 proof.

This does not authorize training. The existing eager V2 aggregate and preflight
must not consume this handoff as if it had their source identity.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_bounded_warm_capture_v1 import audit as audit_full
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import (DATA, OUT, FRESH, PLAN_SHA, archived_frame_metadata,
    phase_artifact, sequence_inputs, validate_plan)
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.run_crossfit_warm_video_study_v2 import checked_audits, certified_seq6, source_hashes
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_bounded_loader_gate_v2 import verify_gate
from scripts.warm_crossfit_video_contract_v1 import read_bound, require

PROTOCOL='protocols/BOUNDED_WARM_FULL_CAPTURE_V1.md'


def sources():
    paths=set(local_import_closure(ROOT,('scripts/verify_bounded_crossfit_handoff_v1.py',)));paths.add(PROTOCOL)
    return {p:digest(ROOT/p) for p in sorted(paths)}


def verify_full_proof(path,sequence,certificate,short_gate,cert_sha,gate_sha):
    path=Path(path).resolve(strict=True);require(path.is_relative_to(OUT.resolve(strict=True)) and
        path.name=='audit.json' and path.parent.parent.name==sequence,'Invalid bounded full audit')
    root=path.parent.parent.parent;sha=phase_artifact(root,f'{sequence}_audit',path);saved=read_bound(path,sha)
    require(saved.get('passed') is True and saved.get('sequence')==sequence and
            saved.get('certificate_sha256')==cert_sha and saved.get('short_gate_sha256')==gate_sha and
            saved.get('label_guard',{}).get('open_attempts')==0,'Bounded full proof differs')
    reconstructed=audit_full(root,sequence,certificate,short_gate)
    require({k:v for k,v in saved.items() if k!='label_guard'}==reconstructed and digest(path)==sha,
            'Bounded full proof reconstruction differs')
    return root,saved,sha


def build(certificate_path,short_gate_path,seq6_path,seq11_path,prior_audits):
    plan=validate_plan();certificate,cert_sha=verify_certificate(certificate_path)
    gate,gate_sha=verify_gate(short_gate_path);require(gate['certificate_sha256']==cert_sha,'Short gate differs')
    root6,proof6,sha6=verify_full_proof(seq6_path,'seq6',certificate_path,short_gate_path,cert_sha,gate_sha)
    root11,proof11,sha11=verify_full_proof(seq11_path,'seq11',certificate_path,short_gate_path,cert_sha,gate_sha)
    require(proof11.get('seq6_proof_sha256')==sha6,'Seq11 does not bind the passed full seq6 proof')
    eager_sources=source_hashes(); accepted=checked_audits(prior_audits,plan,eager_sources,certificate_path)
    require(set(accepted)==set(FRESH)-{'seq6','seq11'},'Eager partial sequence coverage differs')
    certified_seq6(certificate,plan)
    observations=[]; audits={}
    for seq in FRESH:
        sequence_inputs(plan,seq)
        if seq=='seq6':
            capture_file=Path(certificate['seq6_gate_capture']['receipt_path'])
            capture_sha=certificate['seq6_gate_capture']['receipt_sha256']
            reference=certificate['seq6_historical_reference'];ref_path=reference['reference_path'];ref_sha=reference['reference_sha256']
            audit_path=Path(certificate['warm_gate_path']);audit_sha=certificate['warm_gate_sha256']
            reference_kind='historical_warm_reference'; bounded_side=None
        elif seq=='seq11':
            capture_file=root11/seq/'capture/receipt.json';capture_sha=phase_artifact(root11,f'{seq}_capture',capture_file)
            ref_path=str(root11/seq/'reference/receipt.json');ref_sha=phase_artifact(root11,f'{seq}_reference',ref_path)
            audit_path=Path(seq11_path).resolve();audit_sha=sha11;reference_kind='matched_bounded_warm_reference'
            bounded_side=root11/seq/'capture/bounded_receipt.json'
            require(digest(bounded_side)==proof11['capture_bounded_sha256'], 'Bounded seq11 sidecar differs')
        else:
            batch=accepted[seq];capture_file=batch/seq/'capture/receipt.json'
            capture_sha=phase_artifact(batch,f'{seq}_capture',capture_file)
            ref_path=str(batch/seq/'reference/receipt.json');ref_sha=phase_artifact(batch,f'{seq}_reference',ref_path)
            audit_path=batch/seq/'audit/audit.json';audit_sha=phase_artifact(batch,f'{seq}_audit',audit_path)
            reference_kind='matched_warm_reference';bounded_side=None
        capture=read_bound(capture_file,capture_sha);metadata=archived_frame_metadata(plan,seq)
        audits[seq]={'path':str(audit_path),'sha256':audit_sha}
        for item in capture['observations']:
            ordinal=item['frame'];source=plan['source_frames'][seq][ordinal]
            for kind in ('feature','snapshot'):
                require(digest(item[kind+'_file'])==item[kind+'_sha256'],'Handoff observation changed')
            row={'sequence':seq,'frame':ordinal,'view':plan['prompts'][seq]['view'],
                'roi_xyxy':plan['prompts'][seq].get('prompt_provenance',plan['prompts'][seq])['roi_xyxy'],
                'source_frame_index':source['source_frame_index'],'image_sha256':source['image_sha256'],
                'label_metadata_path':metadata['path'],'label_metadata_sha256':metadata['sha256'],
                'label_metadata_locator':ordinal,'reference_kind':reference_kind,'reference_receipt_path':str(ref_path),
                'reference_receipt_sha256':ref_sha,'capture_receipt_path':str(capture_file),
                'capture_receipt_sha256':capture_sha,'sequence_audit_path':str(audit_path),'sequence_audit_sha256':audit_sha,
                'feature_file':item['feature_file'],'feature_sha256':item['feature_sha256'],
                'snapshot_file':item['snapshot_file'],'snapshot_sha256':item['snapshot_sha256']}
            if bounded_side is not None:
                row.update(bounded_sidecar_path=str(bounded_side),bounded_sidecar_sha256=proof11['capture_bounded_sha256'],
                           seq6_proof_path=str(Path(seq6_path).resolve()),seq6_proof_sha256=sha6)
            observations.append(row)
    require(len(observations)==143 and {(x['sequence'],x['frame']) for x in observations}==
            {(x['sequence'],x['frame']) for x in plan['fresh_reads']},'Bounded handoff grid differs')
    return {'passed':True,'format':'bounded_crossfit_handoff_v1','training_authorized':False,
        'scope':'Mixed eager/bounded source evidence; requires a separate bounded training preflight.',
        'index_plan_sha256':PLAN_SHA,'existing_manifest_sha256':plan['existing_manifest_sha256'],
        'existing_feature_pipeline_sha256':plan['feature_pipeline_sha256'],
        'warm_capture_certificate_path':str(Path(certificate_path).resolve()),'warm_capture_certificate_sha256':cert_sha,
        'short_gate_path':str(Path(short_gate_path).resolve()),'short_gate_sha256':gate_sha,
        'seq6_proof_path':str(Path(seq6_path).resolve()),'seq6_proof_sha256':sha6,
        'seq11_proof_path':str(Path(seq11_path).resolve()),'seq11_proof_sha256':sha11,
        'audits':audits,'observations':observations,'source_sha256':sources()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('certificate','short-gate','seq6-audit','seq11-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prior-audit',type=Path,action='append',default=[]);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(),'Invalid handoff output')
    guard=install_mask_open_guard(DATA);result=build(a.certificate,a.short_gate,a.seq6_audit,a.seq11_audit,a.prior_audit)
    require(guard['open_attempts']==0,'Target label opened in handoff');result['label_guard']=dict(guard)
    out.mkdir(parents=True,exist_ok=False);path=out/'handoff.json';path.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'handoff_sha256':digest(path),
        'training_authorized':False,'label_guard':dict(guard)},indent=2)+'\n')
    print(json.dumps({'passed':True,'training_authorized':False,'output':str(out)}))


if __name__=='__main__':main()
