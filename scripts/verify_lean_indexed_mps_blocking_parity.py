#!/usr/bin/env python3
"""Compare indexed-spatial RoPE CPU and MPS Lean on a short real video clip.

No label is read. The inference input is an archived automatic box/point
prompt. This is an engineering diagnostic of the indexed spatial adaptation,
not an author-model result or evidence of segmentation improvement.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import MethodType, SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor, source_identity


def thash(x):
    x = x.detach().cpu().contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'prompt-dir', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--sequence', default='seq4')
    p.add_argument('--end-frame', type=int, default=13)
    a = p.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; do not silently fall back to CPU')
    prompt_path = a.prompt_dir/a.sequence/'prompt.json'
    prompt = json.loads(prompt_path.read_text())
    start = prompt['first_detection_index']
    if not isinstance(start, int) or start <= 0 or a.end_frame <= start:
        raise ValueError('Require nonzero prompt followed by propagation')
    videos = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),
              **load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    frames = videos[a.sequence][:a.end_frame+1]
    metadata = json.loads((a.prompt_dir/a.sequence/'frames.json').read_text())
    if len(frames) != a.end_frame+1 or any(digest(f.image_path) != metadata[i]['image_sha256'] for i,f in enumerate(frames)):
        raise ValueError('Source frame mismatch')
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    repo, rev = source_identity('lean')
    torch.set_num_threads(2)
    cpu, runs = None, {}
    with tempfile.TemporaryDirectory(prefix='lean_mps_parity_') as temp:
        stage_frames(frames, Path(temp))
        for device in ('cpu','mps'):
            model = build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device=device),repo)
            state = model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
            compressions = []
            original = model._compress_memory_entry
            def compress(self, entry):
                old = entry.get('pending_keep_idx')
                indices = None if old is None else old.detach().cpu().tolist()
                result = original(entry)
                if indices is not None:
                    compressions.append({'frame':entry.get('frame_idx'),'retained':len(indices),'indices':indices})
                return result
            model._compress_memory_entry = MethodType(compress, model)
            from sam2.modeling import memory_attention
            observations=[]
            rope = indexed_spatial_rope(model, observations) if device=='cpu' else indexed_spatial_rope_mps(model, observations)
            captured=[]
            if device=='mps':torch.mps.synchronize()
            t0=time.perf_counter()
            transfer = nullcontext() if device=='cpu' else blocking_mps_pointer_position(model)
            with torch.inference_mode(),rope,transfer:
                _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,box=np.asarray(prompt['box'],np.float32))
                if ids!=[1]:raise RuntimeError('Unexpected object')
                if prompt['points'] is not None:
                    _,ids,logits=model.add_new_points_or_box(state,frame_idx=start,obj_id=1,points=np.asarray(prompt['points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
                mask=logits[0,0].cpu().numpy()>0
                if not mask.any():raise RuntimeError('Empty prompt mask')
                model.add_new_mask(state,frame_idx=start,obj_id=1,mask=mask)
                for index,ids,output in model.propagate_in_video(state,start_frame_idx=start,reverse=False):
                    if ids!=[1]:raise RuntimeError('Unexpected object')
                    captured.append((index,output.detach().float().cpu().clone()))
            if device=='mps':torch.mps.synchronize()
            elapsed=time.perf_counter()-t0
            model.reset_state(state)
            del model
            if not compressions:raise RuntimeError(f'{device}: no compression occurred')
            runs[device]={'elapsed_seconds_including_prompt':elapsed,'frames':len(captured),'compressions':compressions,'mask_logits_sha256':[thash(x) for _,x in captured], 'indexed_events':observations}
            if device=='cpu':cpu=captured
            else:
                if [i for i,_ in cpu] != [i for i,_ in captured]:raise RuntimeError('Frame order differs')
                diffs=[]
                for (i,x),(_,y) in zip(cpu,captured):
                    delta=(x-y).abs(); xp=x>0; yp=y>0
                    union=int((xp|yp).sum()); inter=int((xp&yp).sum())
                    diffs.append({'frame':i,'mean_abs_logit_diff':float(delta.mean()),'max_abs_logit_diff':float(delta.max()),
                                  'binary_disagree_pixels':int((xp!=yp).sum()),'binary_union_pixels':union,
                                  'binary_iou':inter/union if union else 1.0})
                runs['comparison']=diffs
    result={'scope':'short nonzero automatic-prompt engineering parity; no labels or quality claim',
            'sequence':a.sequence,'start_frame':start,'end_frame':a.end_frame,'device_modes':'float32 CPU and MPS, experimental indexed spatial RoPE; blocking MPS pointer transfer',
            'caution':'This is not the author baseline; indexed position matches must be independently verified on real frames.',
            'checkpoint_sha256':digest(a.checkpoint),'prompt_sha256':digest(prompt_path),'author_revision':rev,
            'runner_sha256':digest(Path(__file__)),'mps_patch_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py'),
            'torch_version':torch.__version__,'runs':runs}
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'comparison':runs['comparison'],'cpu_seconds':runs['cpu']['elapsed_seconds_including_prompt'],
                      'mps_seconds':runs['mps']['elapsed_seconds_including_prompt']}))


if __name__ == '__main__':main()
