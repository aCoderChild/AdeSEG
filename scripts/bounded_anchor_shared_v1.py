"""Fixed-read raw-candidate study inputs and source identity; no labels read."""
import json
from pathlib import Path
import torch
from modeling.bounded_anchor_raw_refiner import BoundedAnchorRawRefiner
from scripts.memory_token_io import digest
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture
from scripts.train_region_mixed_loss_v1 import dependency_hashes as base_dependencies, mixed_manifest
ROOT=Path(__file__).resolve().parents[1]
ARMS=('raw','zero','spatial_shift')
LOSSES=('bce','robust_bce')
SEED=20261114
EPOCHS=10
LOSS_FORMULAS={'bce':'mean(BCEWithLogits) + soft Dice','robust_bce':'mean(10 * log1p(pixel_BCEWithLogits / 10)) + soft Dice'}

def dependency_hashes():
    result=base_dependencies()
    for name in ('scripts/train_region_mixed_loss_v1.py','modeling/foreground_conditioned_refiner.py','modeling/multilevel_contrast_refiner.py',
                 'modeling/raw_multimask_refiner.py','modeling/bounded_anchor_raw_refiner.py','scripts/bounded_anchor_shared_v1.py',
                 'scripts/audit_bounded_anchor_v1.py','scripts/train_bounded_anchor_study_v1.py',
                 'scripts/score_bounded_anchor_study_v1.py','scripts/capture_raw_multimask_mixed_v1.py','scripts/audit_raw_multimask_mixed_v1.py'):
        result[name]=digest(ROOT/name)
    return result

def load_manifests(folder,raw_audit):
    manifest,mhash=mixed_manifest(folder)
    path=raw_audit/'manifest.json';raw=json.loads(path.read_text());done=json.loads((raw_audit/'completion.json').read_text())
    if (not done.get('passed') or done.get('manifest_sha256')!=digest(path) or not raw.get('passed')
        or raw.get('manifest_sha256')!=mhash or raw.get('validation_labels_opened') is not False
        or raw.get('runner_sha256')!=digest(ROOT/'scripts/audit_raw_multimask_mixed_v1.py')):
        raise ValueError('Frozen raw candidate audit differs')
    key=lambda x:(x['split'],x['sequence'],x['frame'],x['capture_device'])
    samples={key(x):x for x in manifest['samples']}; entries={key(x):x for x in raw['rows']}
    if len(samples)!=49 or len(entries)!=49 or set(samples)!=set(entries) or sum(x['split']=='train' for x in samples.values())!=39:
        raise ValueError('Raw candidate coverage differs')
    for k,item in samples.items():
        row=entries[k]
        if row['snapshot_sha256']!=item['snapshot_sha256'] or digest(row['raw_file'])!=row['raw_sha256']:
            raise ValueError('Raw candidate or snapshot changed')
    return manifest,mhash,entries,digest(path)

def inputs(item,entry,device):
    if entry['snapshot_sha256']!=item['snapshot_sha256'] or digest(entry['raw_file'])!=entry['raw_sha256']:
        raise ValueError('Raw candidate identity changed')
    early,deep,logits,shape=read_capture(item)
    blob=torch.load(entry['raw_file'],map_location='cpu',weights_only=True)
    candidates,ious=blob['raw_candidates'],blob['predicted_ious']
    if candidates.shape!=(logits.shape[0],3,*logits.shape[-2:]) or ious.shape!=(logits.shape[0],3):
        raise ValueError('Raw decoder geometry differs')
    return tuple(x.to(device) for x in (early,deep,logits,candidates,ious)),shape

def entry_for(entries,item):
    return entries[(item['split'],item['sequence'],item['frame'],item['capture_device'])]

def model_for(arm):
    if arm not in ARMS:raise ValueError('Unknown raw candidate control')
    return BoundedAnchorRawRefiner(raw_mode=arm)


def verify_anchor_audit(folder,mhash,rhash):
    path=folder/'audit.json';result=json.loads(path.read_text());done=json.loads((folder/'completion.json').read_text())
    if (not result.get('passed') or not done.get('passed') or done.get('audit_sha256')!=digest(path)
        or result.get('source_sha256')!=digest(ROOT/'scripts/audit_bounded_anchor_v1.py')
        or result.get('manifest_sha256')!=mhash or result.get('raw_manifest_sha256')!=rhash
        or result.get('scale')!=16.0 or len(result.get('rows',[]))!=49):
        raise ValueError('Bounded anchor audit changed')
    return digest(path)
