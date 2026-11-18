#!/usr/bin/env python3
"""Replay automatic Lean with only CPU rotary key-frequency ordering corrected."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest, stage_frames  # noqa: E402
from modeling.early_feature_prompt import early_feature_points  # noqa: E402
from modeling.lean_multiscale_anchor import install_multiscale_anchor  # noqa: E402
from modeling.lean_rope_cpu_repeat_fix import corrected_cpu_key_frequency_repeat  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity  # noqa: E402

PROTOCOL = ROOT / 'protocols/lean_cpu_rope_repeat_fix_v1.json'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'


def read(path): return json.loads(path.read_text())


@contextmanager
def observe_frequency_repeat(module, current_frame, trace, *, check_correct):
    original = module._repeat_freqs_for_k
    checked = set()

    def observe(freqs, repeats):
        output = original(freqs, repeats)
        if repeats > 1:
            key = (int(freqs.shape[0]), int(repeats), str(freqs.dtype))
            trace.append({'frame': current_frame[0], 'base_rows': key[0], 'repeat': key[1], 'dtype': key[2]})
            if check_correct and key not in checked:
                if not torch.equal(output, freqs.repeat(repeats, 1)):
                    raise ValueError('Corrected CPU frequencies differ from CUDA-branch expression')
                checked.add(key)
        return output

    module._repeat_freqs_for_k = observe
    try:
        yield
    finally:
        module._repeat_freqs_for_k = original


def checked_gate(path, expected_identity, expected_mode):
    setup, done = read(path / 'setup.json'), read(path / 'completion.json')
    if (setup['mode'] != expected_mode or setup['sequence'] != 'seq3' or setup['identity'] != expected_identity
        or not done['passed'] or digest(path / 'records.json') != done['records_sha256']):
        raise ValueError(f'Invalid engineering gate: {path}')
    record = read(path / 'records.json')
    if (digest(path / 'frequency_trace.json') != record['frequency_trace_sha256']
        or digest(path / 'selections.json') != record['selections_sha256']
        or not read(path / 'parameter_integrity_after.json')['unchanged']):
        raise ValueError('Engineering gate receipts changed')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','merged','historical','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--mode', choices=('replay','fixed_gate','fixed'), required=True)
    p.add_argument('--sequence', required=True)
    for name in ('replay-a','replay-b','fixed-a','fixed-b'):
        p.add_argument('--'+name, type=Path)
    a = p.parse_args()
    protocol = read(PROTOCOL)
    if a.sequence not in protocol['cohort'] or (a.mode != 'fixed' and a.sequence != 'seq3'):
        raise ValueError('Sequence outside requested scope')
    merged = read(a.merged / 'completion.json')
    if not merged['passed'] or digest(a.merged/'results.json') != merged['results_sha256'] or digest(a.historical/'comparison.json') != merged['historical_comparison_sha256']:
        raise ValueError('Reference comparison incomplete or changed')
    source = Path(read(a.merged/'locations.json')[a.sequence]); folder = source / a.sequence
    old_setup, old_done, audit = read(source/'setup.json'), read(source/'completion.json'), read(source/'independent_audit.json')
    if not old_done['passed'] or not audit['passed'] or digest(source/'records.json') != old_done['records_sha256'] or audit['records_sha256'] != old_done['records_sha256']:
        raise ValueError('Old run incomplete')
    sig = old_setup['signature']
    pinned = {'early_module_sha256': ROOT/'modeling/early_feature_prompt.py', 'prompt_module_sha256': ROOT/'modeling/prompt_feature_refinement.py',
              'memory_module_sha256': ROOT/'modeling/lean_multiscale_anchor.py', 'lean_runner_sha256': ROOT/'scripts/run_author_baseline.py',
              'runner_sha256': ROOT/'scripts/run_lean_auto_multiscale.py', 'manifest_sha256': MANIFEST,
              'bypass_sha256': ROOT/'external/Lean-SAM2/checkpoints/bypass_bottleneck_base.pth',
              'lean_config_sha256': ROOT/'external/Lean-SAM2/configs/lean_sam2/lean_sam2.yaml'}
    if (digest(a.checkpoint) != sig['checkpoint_sha256'] or any(digest(path) != sig[key] for key,path in pinned.items())
        or torch.__version__ != sig['torch'] or np.__version__ != sig['numpy'] or sys.version != sig['python']):
        raise ValueError('Model, runtime, input or source differs from old run')
    identity = {'runner_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PROTOCOL),
                'fix_sha256': digest(ROOT/'modeling/lean_rope_cpu_repeat_fix.py'),
                'checkpoint_sha256': digest(a.checkpoint), 'merged_results_sha256': merged['results_sha256'], 'source_signature': sig}
    if a.mode != 'replay':
        if a.replay_a is None or a.replay_b is None: raise ValueError('Two baseline replay gates required')
        replays = [checked_gate(g, identity, 'replay') for g in (a.replay_a,a.replay_b)]
        if replays[0] != replays[1]: raise ValueError('Baseline replays differ')
    if a.mode == 'fixed':
        if a.fixed_a is None or a.fixed_b is None: raise ValueError('Two fixed engineering gates required')
        gates = [checked_gate(g, identity, 'fixed_gate') for g in (a.fixed_a,a.fixed_b)]
        if gates[0] != gates[1]: raise ValueError('Fixed engineering replays differ')
    old_record = read(source/'records.json')[0]
    for key, filename in [('prompt_sha256','prompt.json'),('frames_sha256','frames.json'),('selections_sha256','selections.json')]:
        if digest(folder/filename) != old_record[key]: raise ValueError('Old receipt changed')
    prompt, old_rows = read(folder/'prompt.json'), read(folder/'frames.json')
    index = prompt['first_detection_index']
    if index is None: raise ValueError('No original automatic detection')
    videos = {**load_manifest(MANIFEST,'dev'), **load_manifest(MANIFEST,'test')}; frames = videos[a.sequence]
    if len(frames) != len(old_rows): raise ValueError('Frame count mismatch')
    for frame,row in zip(frames,old_rows):
        if digest(frame.image_path) != row['image_sha256']: raise ValueError('Input image changed')
    out = validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
    (out/'setup.json').write_text(json.dumps({'mode':a.mode,'sequence':a.sequence,'identity':identity,
                                             'original_records_sha256':old_done['records_sha256']},indent=2)+'\n')
    torch.set_num_threads(2)
    repo, revision = source_identity('lean')
    if revision != sig['revision']: raise ValueError('Lean source differs')
    model = build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),repo)
    if model.mem_keep_ratio != .25: raise ValueError('Lean token budget differs')
    before = parameter_fingerprints(model)
    selections = []; memory_events = install_multiscale_anchor(model,.5,selection_events=selections)
    from sam2.modeling import memory_attention as lean_memory_attention
    initial_helper = lean_memory_attention._repeat_freqs_for_k
    trace, current = [], [None]
    for i in range(index):
        with Image.open(frames[i].image_path) as im: w,h=im.size
        Image.fromarray(np.zeros((h,w),np.uint8)).save(out/f'{i:06d}.png')
    patch = nullcontext() if a.mode == 'replay' else corrected_cpu_key_frequency_repeat(lean_memory_attention)
    with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as tmp:
        stage_frames(frames,Path(tmp))
        staged=sorted(Path(tmp).glob('*.jpg'),key=lambda path:int(path.stem))
        if len(staged)!=len(frames) or any(p.resolve()!=f.image_path.resolve() for p,f in zip(staged,frames)):
            raise ValueError('Frame order differs')
        state=model.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        try:
            with torch.inference_mode(), patch, observe_frequency_repeat(lean_memory_attention,current,trace,check_correct=a.mode!='replay'):
                box=np.asarray(prompt['box'],np.float32)
                _,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,box=box)
                if ids != [1]: raise ValueError('Unexpected object ID')
                provisional=logits[0,0].detach().cpu().numpy()>0
                if digest(folder/'prompt_provisional.png') != prompt['provisional_mask_sha256'] or not np.array_equal(provisional,np.asarray(Image.open(folder/'prompt_provisional.png'))>0):
                    raise ValueError('Box prompt does not replay')
                _,backbone,_,_,_=model._get_image_feature(state,index,batch_size=1)
                points,status=early_feature_points(backbone,provisional,box,seed=20261114)
                if status != prompt['point_status'] or (None if points is None else points['feature_points']) != prompt['points']:
                    raise ValueError('Feature points do not replay')
                if points is not None:
                    _,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,
                        points=np.asarray(points['feature_points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
                initial=logits[0,0].detach().cpu().numpy()>0
                if digest(folder/'prompt_refined.png') != prompt['refined_mask_sha256'] or not np.array_equal(initial,np.asarray(Image.open(folder/'prompt_refined.png'))>0):
                    raise ValueError('Refined prompt does not replay')
                if initial.any(): model.add_new_mask(state,frame_idx=index,obj_id=1,mask=initial)
                generator=model.propagate_in_video(state,start_frame_idx=index,reverse=False)
                for expected in range(index,len(frames)):
                    current[0]=expected
                    actual,ids,output=next(generator)
                    if actual != expected or ids != [1]: raise ValueError('Propagation order or object differs')
                    Image.fromarray((output[0,0].detach().cpu().numpy()>0).astype(np.uint8)*255).save(out/f'{actual:06d}.png')
                try:
                    next(generator)
                    raise ValueError('Unexpected extra frame')
                except StopIteration:
                    pass
        finally:
            model.reset_state(state)
    if lean_memory_attention._repeat_freqs_for_k is not initial_helper:
        raise ValueError('CPU helper patch leaked outside its scope')
    assert_parameter_integrity(model,before,out,'after')
    masks=[digest(out/f'{i:06d}.png') for i in range(len(frames))]
    if not trace: raise ValueError('No affected key-frequency calls were observed')
    first_affected=min(x['frame'] for x in trace)
    if any(masks[i] != old_rows[i]['mask_sha256'] for i in range(first_affected)):
        raise ValueError('Masks differ before the first corrected operation')
    if a.mode == 'replay' and (masks != [x['mask_sha256'] for x in old_rows] or selections != read(folder/'selections.json')):
        raise ValueError('Unmodified replay differs from archived masks or selections')
    (out/'frequency_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    (out/'selections.json').write_text(json.dumps(selections,indent=2)+'\n')
    (out/'memory_events.json').write_text(json.dumps(memory_events,indent=2)+'\n')
    if a.mode == 'fixed':
        scored=[]
        # Labels are opened only after all video predictions have been saved.
        for i,frame in enumerate(frames):
            if digest(frame.mask_path) != old_rows[i]['label_sha256']: raise ValueError('Label changed')
            truth=load_label_mask(frame.mask_path,[1])==1
            pred=np.asarray(Image.open(out/f'{i:06d}.png'))>0
            if pred.shape != truth.shape: raise ValueError('Mask shape mismatch')
            den=int(pred.sum()+truth.sum()); present=bool(truth.any())
            scored.append({'frame':i,'present':present,'dice':2*int((pred & truth).sum())/den if den else 1.,
                           'false_alarm':bool(pred.any()) if not present else None,'mask_sha256':masks[i]})
        (out/'frames.json').write_text(json.dumps(scored,indent=2)+'\n')
    record={'sequence':a.sequence,'frames':len(frames),'first_detection_index':index,'first_affected_frame':first_affected,
            'affected_calls':len(trace),'repeat_shapes':dict(Counter(f"{x['base_rows']}x{x['repeat']}" for x in trace)),
            'mask_sha256':masks,'selections_sha256':digest(out/'selections.json'),
            'frequency_trace_sha256':digest(out/'frequency_trace.json'), 'memory_events_sha256':digest(out/'memory_events.json'),
            'original_records_sha256':old_done['records_sha256'], 'frames_sha256':digest(out/'frames.json') if a.mode=='fixed' else None}
    (out/'records.json').write_text(json.dumps(record,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(out/'records.json')},indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k != 'mask_sha256'},indent=2))


if __name__=='__main__': main()
