#!/usr/bin/env python3
"""Verify two frozen same-device decoder replays for all 49 mixed reads.

This routine never opens labels. It does not authorize video inference or
assert that the previously inspected validation cohort is independent.
"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_lean_raw_decoder_train_reads import select_raw_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_offline_memory_ablation import equal
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture
from scripts.train_region_mixed_loss_v1 import mixed_manifest


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-preflight','cpu-a','cpu-b','mps-a','mps-b','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();manifest,mhash=mixed_manifest(a.data_preflight)
    roots={'cpu':(a.cpu_a,a.cpu_b),'mps':(a.mps_a,a.mps_b)}
    expected={(x['split'],x['sequence'],x['frame']):x for x in manifest['samples']}
    captured={};states=[];receipt_hashes={}
    for device,(first,second) in roots.items():
        receipts=[read(x/'receipt.json') for x in (first,second)]
        for folder,r in zip((first,second),receipts,strict=True):
            if (not r['passed'] or r['device']!=device or r['manifest_sha256']!=mhash
                    or r['runner_sha256']!=digest(ROOT/'scripts/capture_raw_multimask_mixed_v1.py')
                    or r['selector_source_sha256']!=digest(ROOT/'scripts/diagnose_lean_raw_decoder_train_reads.py')
                    or r['protocol_sha256']!=digest(ROOT/'protocols/RAW_MULTIMASK_REFINER_BRANCH_V1.md')
                    or r['checkpoint_sha256']!=manifest['checkpoint_sha256'] or r['validation_labels_opened'] is not False):
                raise ValueError('Replay receipt identity differs')
            if device=='mps' and (r['fallback']!='0' or r['fast_math']!='0'):
                raise ValueError('MPS fallback or fast math differs')
            if len(r['rows'])!=(29 if device=='cpu' else 20):raise ValueError('Per-device replay coverage differs')
            states.append(r['model_state_sha256']);receipt_hashes[str(folder)]=digest(folder/'receipt.json')
        if receipts[0]['model_state_sha256']!=receipts[1]['model_state_sha256']:
            raise ValueError('Repeated model state differs')
        pairs=[{(r['split'],r['sequence'],r['frame']):r for r in x['rows']} for x in receipts]
        keys={key for key,item in expected.items() if item['capture_device']==device}
        if any(set(pair)!=keys or len(pair)!=len(receipt['rows']) for pair,receipt in zip(pairs,receipts,strict=True)):
            raise ValueError('Repeated read coverage differs')
        for key in sorted(keys):
            item=expected[key];blobs=[]
            for folder,pair in zip((first,second),pairs,strict=True):
                row=pair[key];path=Path(row['raw_file'])
                if (not path.resolve().is_relative_to(folder.resolve()) or digest(path)!=row['raw_sha256']
                        or row['capture_device']!=device or row['snapshot_sha256']!=item['snapshot_sha256']):
                    raise ValueError('Raw tensor or snapshot identity differs')
                blobs.append(torch.load(path,map_location='cpu',weights_only=True))
            if not equal(blobs[0],blobs[1]):raise ValueError('Same-device raw decoder replay differs')
            raw=blobs[0];snapshot=torch.load(item['snapshot'],map_location='cpu',weights_only=True);read_capture(item)
            if (raw['sequence']!=key[1] or raw['frame']!=key[2] or raw['raw_candidates'].shape!=(1,3,256,256)
                    or raw['predicted_ious'].shape!=(1,3) or not all(torch.isfinite(raw[k]).all() for k in ('raw_candidates','predicted_ious','object_score_logits','native_selected','native_pointer'))
                    or not torch.equal(raw['predicted_ious'],snapshot['decoder_outputs'][2])
                    or not torch.equal(raw['native_selected'],snapshot['decoder_outputs'][3])
                    or not torch.equal(raw['native_pointer'],snapshot['decoder_outputs'][5])
                    or not torch.equal(raw['object_score_logits'],snapshot['decoder_outputs'][6])):
                raise ValueError('Raw candidates differ from saved native head state')
            selected,index=select_raw_mask(raw['raw_candidates'],raw['predicted_ious'],True)
            if not torch.equal(index,raw['selected_index']):raise ValueError('Author IoU-selected index differs')
            score=float(raw['object_score_logits'][0,0])
            if (score>0 and not torch.equal(selected,raw['native_selected'])) or (score<=0 and not torch.all(raw['native_selected']==-1024)):
                raise ValueError('Native score/mask suppression consistency differs')
            captured[key]={'split':key[0],'sequence':key[1],'frame':key[2],
                           'capture_device':device,'snapshot_sha256':item['snapshot_sha256'],
                           'raw_file':pairs[0][key]['raw_file'],'raw_sha256':pairs[0][key]['raw_sha256'],
                           'repeat_raw_sha256':pairs[1][key]['raw_sha256'],'selected_index':int(index[0]),'object_score':score}
    if set(captured)!=set(expected) or len(set(states))!=1:raise ValueError('Overall coverage or model state differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'scope':'Label-free repeated decoder replay on frozen reads only. No video inference or no-object bypass.',
            'manifest_sha256':mhash,'checkpoint_sha256':manifest['checkpoint_sha256'],'model_state_sha256':states[0],
            'training_count':sum(k[0]=='train' for k in captured),'validation_count':sum(k[0]=='validation' for k in captured),
            'validation_labels_opened':False,'runner_sha256':digest(Path(__file__)),'receipt_hashes':receipt_hashes,
            'rows':[captured[k] for k in sorted(captured)]}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'manifest_sha256':digest(out/'manifest.json')},indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','training_count','validation_count','validation_labels_opened')}))


if __name__=='__main__':main()
