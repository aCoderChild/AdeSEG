import json
from pathlib import Path

import pytest
from scripts import verify_bounded_crossfit_handoff_v1 as handoff
from scripts.memory_token_io import digest


def test_handoff_requires_seq6_proof_and_seq11_sidecar(tmp_path,monkeypatch):
    monkeypatch.setattr(handoff,'FRESH',('seq6','seq11'))
    shared=tmp_path/'tensor.pt';shared.write_bytes(b'tensor');sha=digest(shared)
    def observations(count):
        return [{'frame':i,'feature_file':str(shared),'feature_sha256':sha,
                 'snapshot_file':str(shared),'snapshot_sha256':sha} for i in range(count)]
    seq6=tmp_path/'seq6_capture.json';seq6.write_text(json.dumps({'observations':observations(72)}))
    root11=tmp_path/'bounded11';cap=root11/'seq11/capture';ref=root11/'seq11/reference'
    cap.mkdir(parents=True);ref.mkdir(parents=True)
    capture=cap/'receipt.json';capture.write_text(json.dumps({'observations':observations(71)}))
    (ref/'receipt.json').write_text('{}');side=cap/'bounded_receipt.json';side.write_text('bounded')
    proof6=tmp_path/'seq6-audit.json';proof6.write_text('{}');proof11=tmp_path/'seq11-audit.json';proof11.write_text('{}')
    source_frames={seq:[{'source_frame_index':i,'image_sha256':'image'} for i in range(count)]
                   for seq,count in [('seq6',72),('seq11',71)]}
    fresh=[{'sequence':seq,'frame':i} for seq,count in [('seq6',72),('seq11',71)] for i in range(count)]
    prompts={seq:{'view':{'kind':'full_source_frame'},'roi_xyxy':[0,0,1,1]} for seq in ('seq6','seq11')}
    plan={'fresh_reads':fresh,'source_frames':source_frames,'prompts':prompts,
          'existing_manifest_sha256':'old','feature_pipeline_sha256':'pipeline'}
    cert={'seq6_gate_capture':{'receipt_path':str(seq6),'receipt_sha256':digest(seq6)},
          'seq6_historical_reference':{'reference_path':str(ref/'receipt.json'),
                                       'reference_sha256':digest(ref/'receipt.json')},
          'warm_gate_path':str(proof6),'warm_gate_sha256':digest(proof6)}
    monkeypatch.setattr(handoff,'validate_plan',lambda:plan)
    monkeypatch.setattr(handoff,'verify_certificate',lambda path:(cert,'cert'))
    monkeypatch.setattr(handoff,'verify_gate',lambda path:({'certificate_sha256':'cert'},'gate'))
    def full(path,sequence,*args):
        if sequence=='seq6':return tmp_path,{},'seq6-proof'
        return root11,{'seq6_proof_sha256':'seq6-proof','capture_bounded_sha256':digest(side)},'seq11-proof'
    monkeypatch.setattr(handoff,'verify_full_proof',full)
    monkeypatch.setattr(handoff,'source_hashes',lambda:{})
    monkeypatch.setattr(handoff,'checked_audits',lambda *args:{})
    monkeypatch.setattr(handoff,'certified_seq6',lambda *args:None)
    monkeypatch.setattr(handoff,'sequence_inputs',lambda *args:None)
    monkeypatch.setattr(handoff,'archived_frame_metadata',lambda *args:{'path':'metadata','sha256':'metadata-sha'})
    monkeypatch.setattr(handoff,'phase_artifact',lambda root,name,path:digest(path))
    monkeypatch.setattr(handoff,'sources',lambda:{})
    result=handoff.build('cert','gate',proof6,proof11,[])
    assert result['training_authorized'] is False and len(result['observations'])==143
    bounded=[x for x in result['observations'] if x['sequence']=='seq11']
    assert len(bounded)==71 and all(x['bounded_sidecar_sha256']==digest(side) for x in bounded)
    monkeypatch.setattr(handoff,'verify_full_proof',lambda path,sequence,*args:
        (tmp_path,{},'seq6-proof') if sequence=='seq6' else
        (root11,{'seq6_proof_sha256':'wrong','capture_bounded_sha256':digest(side)},'seq11-proof'))
    with pytest.raises(ValueError,match='does not bind'):handoff.build('cert','gate',proof6,proof11,[])
    monkeypatch.setattr(handoff,'verify_full_proof',lambda path,sequence,*args:
        (tmp_path,{},'seq6-proof') if sequence=='seq6' else
        (root11,{'seq6_proof_sha256':'seq6-proof','capture_bounded_sha256':'bad'},'seq11-proof'))
    with pytest.raises(ValueError,match='sidecar differs'):handoff.build('cert','gate',proof6,proof11,[])
