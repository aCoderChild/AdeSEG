"""Gate and capture official SAM2 native trajectories and stage features."""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.native_history_cache import bank_payload, load_bank
from modeling.native_stage_capture import NativeStageCapture
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor, tensor_hash
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_regional_refinement import load, save_tensor, write
from scripts.run_soft_role_affinity_cv import score_frame, summarize

PROTOCOL = 'protocols/encoder_decoder_bottleneck_v1.json'


def capture_sequence(protocol, provenance, frames, sequence, folder, *, observe, reference=None):
    folder.mkdir(parents=True, exist_ok=False)
    repo, revision = activate_author_repo()
    model = build_author_predictor(SimpleNamespace(method='sam2', device='cpu',
        checkpoint=Path(protocol['checkpoint']), model_config=protocol['model_config']), repo)
    weights = parameter_digest(model)
    selected = {c['frame'] for c in protocol['cases'] if c['sequence'] == sequence}
    rows, artifacts = [], []
    with tempfile.TemporaryDirectory(prefix='official_native_', dir=folder) as directory, torch.inference_mode():
        state, prompt = prepare(model, frames, Path(directory))
        manager = NativeStageCapture(model) if observe else nullcontext()
        try:
            with manager as observer:
                trajectory = iter(model.propagate_in_video(state, start_frame_idx=prompt))
                for expected in range(prompt, len(frames)):
                    if observe:
                        observer.begin(expected in selected)
                    frame, ids, logits = next(trajectory)
                    if frame != expected or ids != [1]:
                        raise RuntimeError('Native frame or object order differs')
                    if frame == prompt:
                        continue
                    stored = state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                    row = {**score_frame(logits, frames, frame),
                        'stored_sha256': fingerprint(stored)['sha256']}
                    if reference is not None and row != reference[len(rows)]:
                        raise RuntimeError(f'Observer changed native trajectory: {sequence}/{frame}')
                    if observe:
                        if [r['layer'] for r in observer.layers] != [0, 1, 2, 3]:
                            raise RuntimeError('Missing native attention layer')
                        if any((r['keys']-r['pointer_keys']) % 4096 for r in observer.layers):
                            raise RuntimeError('Invalid native spatial-key grid')
                        if frame in selected:
                            tensors = observer.tensors
                            required = {'current_image', 'memory_conditioned', 'high_res_features',
                                        'raw_masks', 'predicted_ious', 'mask_tokens', 'object_score'}
                            if set(tensors) != required:
                                raise RuntimeError('Incomplete native stage capture')
                            values = [v for v in tensors.values() if isinstance(v,torch.Tensor)] + tensors['high_res_features']
                            if any(not torch.isfinite(v).all() for v in values):
                                raise RuntimeError('Nonfinite native feature or decoder snapshot')
                            tensors.update(pred_masks=stored['pred_masks'].detach().clone(),
                                target=torch.from_numpy((load_label_mask(frames[frame].mask_path, [1]) == 1).copy()))
                            if not torch.equal(tensors['object_score'], stored['object_score_logits']):
                                raise RuntimeError('Captured decoder score differs')
                            path = folder / f'frame_{frame}.pt'
                            save_tensor(tensors, path)
                            restored = load(path, digest(path))
                            if fingerprint(restored)['sha256'] != fingerprint(tensors)['sha256']:
                                raise RuntimeError('Stage serialization changed values/layout')
                            artifacts.append({'frame':frame, 'path':str(path), 'sha256':digest(path),
                                'layers':observer.layers, 'stage_fingerprint':fingerprint(tensors)['sha256']})
                    rows.append(row)
                    write(folder.parent / 'progress.json', {'sequence':sequence,'frame':frame,'observe':observe})
            if len(rows) != len(frames)-prompt-1:
                raise RuntimeError('Incomplete native trajectory')
            if observe:
                if {r['frame'] for r in artifacts} != selected:
                    raise RuntimeError('Fixed feature-case coverage differs')
                payload = bank_payload(state)
                save_tensor(payload, folder / 'bank.pt')
                receipt = {**provenance,'sequence':sequence,'object_id':1,'num_frames':len(frames),
                    'prompt':prompt,'predictions':[{'frame':r['frame'],'sha256':r['prediction_sha256']} for r in rows],
                    'native_frames_checked':len(rows),'payload_sha256':fingerprint(payload)['sha256'],
                    'bank_sha256':digest(folder/'bank.pt'),'bank_bytes':(folder/'bank.pt').stat().st_size}
                load_bank(folder/'bank.pt', receipt, **provenance)
                write(folder/'receipt.json',receipt)
        finally:
            model.reset_state(state)
    if parameter_digest(model) != weights:
        raise RuntimeError('Native weights changed during capture')
    write(folder/'frames.json', rows)
    write(folder/'completion.json', {'complete':True,'observe':observe,'sequence':sequence,
        'targets':len(rows),'prompt':prompt,'summary':summarize(rows),'artifacts':artifacts,
        'frames_sha256':digest(folder/'frames.json'),'parameters_unchanged':True,'sam2_revision':revision})
    del model
    gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gate-only',action='store_true')
    parser.add_argument('--gate',type=Path)
    args = parser.parse_args()
    if not args.gate_only and args.gate is None:
        parser.error('The cohort capture requires --gate')
    out = validate_output(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    protocol = json.loads((ROOT/PROTOCOL).read_text())
    prior = json.loads((Path(protocol['prior_diagnostic'])/'full/completion.json').read_text())
    sources = {**prior['sources'],**{p:digest(ROOT/p) for p in
        [PROTOCOL,'scripts/capture_official_native_stages.py','modeling/native_stage_capture.py',
         'tests/test_native_stage_capture.py','modeling/decoder_adaptation.py',
         'tests/test_decoder_adaptation.py','external/sam2/training/trainer.py',
         'external/sam2/training/utils/distributed.py',
         'external/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml']}}
    if any(digest(ROOT/p)!=h for p,h in sources.items()):
        raise RuntimeError('Frozen source differs')
    if digest(Path(protocol['checkpoint']))!=protocol['checkpoint_sha256']:
        raise RuntimeError('Official checkpoint differs')
    provenance = {'checkpoint_sha256':protocol['checkpoint_sha256'],
        'manifest_sha256':digest(ROOT/protocol['manifest']),'model_config':protocol['model_config']}
    if provenance['manifest_sha256']!=protocol['manifest_sha256']:
        raise RuntimeError('Manifest differs')
    write(out/'setup.json',{**provenance,'sources':sources})
    videos = {**load_manifest(ROOT/protocol['manifest'],'dev'),
              **load_manifest(ROOT/protocol['manifest'],'test')}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    entries = {}
    if args.gate_only:
        for name in protocol['gate_sequences']:
            plain = capture_sequence(protocol,provenance,videos[name],name,out/'plain'/name,observe=False)
            capture_sequence(protocol,provenance,videos[name],name,out/'observed'/name,
                observe=True,reference=plain)
            entries[name] = str(out/'observed'/name)
    else:
        gate = json.loads((args.gate/'completion.json').read_text())
        if not gate.get('passed') or gate['sources']!=sources or gate['scope']!='gate':
            raise RuntimeError('Official native gate missing or different')
        for name, path in gate['entries'].items():
            folder = Path(path)
            receipt = json.loads((folder/'receipt.json').read_text())
            load_bank(folder/'bank.pt',receipt,**provenance)
            complete = json.loads((folder/'completion.json').read_text())
            if digest(folder/'frames.json')!=complete['frames_sha256']:
                raise RuntimeError('Gate frame records changed')
            if any(digest(Path(r['path']))!=r['sha256'] for r in complete['artifacts']):
                raise RuntimeError('Gate feature records changed')
            entries[name]=path
        for name in protocol['sequences']:
            if name not in entries:
                capture_sequence(protocol,provenance,videos[name],name,out/name,observe=True)
                entries[name]=str(out/name)
    write(out/'completion.json',{'passed':True,'scope':'gate' if args.gate_only else 'cohort',
        'sources':sources,'entries':entries,'checkpoint_sha256':provenance['checkpoint_sha256'],
        'note':'Unchanged official native predictor. Features are observations, not trained outputs.'})


if __name__ == '__main__':
    main()
