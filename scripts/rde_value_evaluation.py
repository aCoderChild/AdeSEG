"""Reuse native autoregressive value evaluation with transactional cache checks.

No teacher cache/GT value target is loaded. Ground truth is read by the existing
metric evaluator only after prediction, apart from SAM2's initial mask prompt.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import resource
import sys

import torch
from modeling.rde_value_repair import MemoryValueRepair,bind_native_reads
from scripts.mavos_value_utils import atomic_json,state_digest
from scripts.memory_token_io import digest,parameter_digest,prompt_index
from scripts.run_mavos_value_contrastive import evaluate_video as author_evaluate
from scripts.run_token_conditioned_key_adapter import _atomic_jsonl,_sequence_source_hashes,_validate_cached_evaluation


def load_adapter(path,training_identity,arm):
    saved=torch.load(path,weights_only=True,map_location='cpu')
    if set(saved)!={'adapter','identity_sha256','arm','frames_seen'}:
        raise RuntimeError('Inference checkpoint contains unexpected training/cache fields')
    if saved['identity_sha256']!=training_identity or saved['arm']!=arm:
        raise RuntimeError('Inference checkpoint training/arm identity differs')
    adapter=MemoryValueRepair();adapter.load_state_dict(saved['adapter'],strict=True)
    return adapter.requires_grad_(False).eval()


def verify_value_rows(rows,adapter_used):
    if not adapter_used:return
    for row in rows:
        read=row.get('memory_read',{});values=row.get('value_calls',[])
        if read.get('frame')!=row['frame'] or read.get('object_id')!=1 or read.get('layers')!=[0,1,2,3]:
            raise RuntimeError('Evaluation frame/object/four-layer ownership differs')
        if read.get('stored_mask_alignment') is not True or read.get('native_positions_verified') is not True:
            raise RuntimeError('Evaluation native memory/position alignment failed')
        if not 0<read.get('spatial_keys',0)<=28672 or read['spatial_keys']%4096:
            raise RuntimeError('Evaluation spatial frame budget differs')
        if [v['layer'] for v in values]!=[0,1,2,3] or any(v['pointer_tail_bit_identical'] is not True for v in values):
            raise RuntimeError('Evaluation pointer/four-layer value coverage differs')
        if any(v['spatial_tokens']!=read['spatial_keys'] or v['pointer_tokens']!=read['pointer_keys'] for v in values):
            raise RuntimeError('Evaluation actual key/value counts differ')


def evaluate_condition(model,frames,adapter,name,out,setup_hash):
    """Commit one fully rescored sequence; refuse partial or corrupt caches."""
    if not frames or any(frame.video_id!=name for frame in frames):
        raise RuntimeError('Evaluation sequence name differs from manifest ownership')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Inference requires frozen eval SAM2')
    if adapter is not None and (adapter.training or any(p.requires_grad for p in adapter.parameters())):
        raise RuntimeError('Inference requires a frozen eval adapter')
    out=Path(out);directory=out/'evaluation'/name
    model_hash=parameter_digest(model)
    heads=None if adapter is None else torch.nn.ModuleList([adapter]*4)
    adapter_hash='native' if heads is None else state_digest(heads.state_dict())
    prompt=prompt_index(frames,{'initialization':'first_labeled_frame'})
    if directory.exists():
        receipt=json.loads((directory/'completion.json').read_text())
        if receipt['passed'] is not True or receipt['setup_core_sha256']!=setup_hash or receipt['adapter_sha256']!=adapter_hash or receipt['sam2_sha256']!=model_hash:
            raise RuntimeError('Evaluation cache checkpoint/setup identity differs')
        if receipt['frames_sha256']!=digest(directory/'frames.jsonl') or receipt['summary_sha256']!=digest(directory/'summary.json'):
            raise RuntimeError('Evaluation cache file checksum differs')
        summary=_validate_cached_evaluation(directory,frames,prompt,adapter_hash,model_hash,setup_hash)
        rows=[json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
        verify_value_rows(rows,heads is not None)
        return summary
    stage=out/'staging'/name
    if stage.exists():raise RuntimeError('Uncommitted evaluation stage exists; preserve and audit it before retry')
    stage.mkdir(parents=True,exist_ok=False)
    inputs=_sequence_source_hashes(frames)
    atomic_json(stage/'input_identity.json',{'inputs':inputs,'setup_core_sha256':setup_hash,
        'adapter_sha256':adapter_hash,'sam2_sha256':model_hash})
    with bind_native_reads(model,adapter) if adapter is not None else nullcontext():
        summary=author_evaluate(model,frames,heads,name,stage,setup_hash)
    staged=stage/'evaluation'/name
    if _sequence_source_hashes(frames)!=inputs:raise RuntimeError('Dataset changed during sequence evaluation')
    rows=[json.loads(line) for line in (staged/'frames.jsonl').read_text().splitlines()]
    for row in rows:
        source=inputs[row['frame']]
        row['image_sha256']=source['image_sha256'];row['ground_truth_sha256']=source['mask_sha256']
        row['source_frame_index']=frames[row['frame']].frame_index
    verify_value_rows(rows,heads is not None)
    summary['input_frames']=inputs
    summary['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    summary['peak_rss_scope']='Process lifetime high-water mark including staging/model; not isolated per-sequence allocation'
    _atomic_jsonl(staged/'frames.jsonl',rows);atomic_json(staged/'summary.json',summary)
    # Reuse the project's strict saved-mask metric validator, not a replacement
    # Dice/IoU/boundary implementation or a relaxed numerical tolerance.
    checked=_validate_cached_evaluation(staged,frames,prompt,adapter_hash,model_hash,setup_hash)
    receipt={'passed':True,'frames_sha256':digest(staged/'frames.jsonl'),'summary_sha256':digest(staged/'summary.json'),
        'setup_core_sha256':setup_hash,'adapter_sha256':adapter_hash,'sam2_sha256':model_hash,
        'saved_masks_independently_rescored':True,'value_read_audited':heads is not None,
        'prediction_logit_hash_scheme':'run_author_baseline.tensor_hash float32 raw bytes; use saved masks for comparisons with native-reference shape-prefixed hashes'}
    atomic_json(staged/'completion.json',receipt)
    directory.parent.mkdir(parents=True,exist_ok=True);staged.rename(directory)
    atomic_json(stage/'committed.json',{'directory':str(directory),'completion_sha256':digest(directory/'completion.json')})
    return checked
