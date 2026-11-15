#!/usr/bin/env python3
"""Replay all frozen validation predictions without reading label masks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.source_group_boundary_controls_shared_v1 import ARMS, EPOCHS, LOSS, PARAMETERS, SEED, dependencies, load_prepared, model_for, read
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture, to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.train_source_group_boundary_controls_v1 import require_parity, require_plan


def check_frozen_arm(folder, arm, manifest, manifest_sha, plan_path, parity_path, plan):
    completion = read(folder/'completion.json'); frozen = read(folder/'frozen_predictions.json'); ck = folder/'checkpoint.pt'
    train_paths, blocked_paths = mask_path_sets(ROOT/'data/polypgen_sequence.jsonl')
    if (not completion.get('passed') or completion.get('arm') != arm or completion.get('manifest_sha256') != manifest_sha
            or completion.get('plan_sha256') != digest(plan_path) or completion.get('parity_sha256') != digest(parity_path)
            or completion.get('checkpoint_sha256') != digest(ck) or frozen.get('checkpoint_sha256') != digest(ck)
            or completion.get('frozen_predictions_sha256') != digest(folder/'frozen_predictions.json')
            or completion.get('initial_state_sha256') != plan['initial_state_sha256']
            or completion.get('training_order') != plan['training_order']
            or completion.get('label_guard', {}).get('blocked_open_attempts') != 0
            or completion.get('label_guard', {}).get('train_paths') != len(train_paths)
            or completion.get('label_guard', {}).get('blocked_paths') != len(blocked_paths)
            or len(completion.get('history', [])) != EPOCHS
            or [x.get('epoch') for x in completion['history']] != list(range(1, EPOCHS+1))
            or any(x.get('samples') != 164 or x.get('sample_order') != plan['training_order'][i]
                   or not np.isfinite(x.get('mean_training_loss', float('nan'))) for i, x in enumerate(completion['history']))):
        raise ValueError(f'{arm}: incomplete training or frozen predictions')
    payload = torch.load(ck, map_location='cpu', weights_only=True)
    expected = {'research_only': True, 'training_steps': 164*EPOCHS, 'study_arm': arm,
                'train_manifest_sha256': manifest_sha, 'backbone_sha256': manifest['checkpoint_sha256'],
                'feature_pipeline_sha256': manifest['feature_pipeline_sha256'], 'plan_sha256': digest(plan_path),
                'parity_sha256': digest(parity_path), 'initial_state_sha256': plan['initial_state_sha256'],
                'dependency_sha256': dependencies(), 'environment': plan['environment'], 'seed': SEED,
                'epochs': EPOCHS, 'parameter_count': PARAMETERS, 'loss': LOSS[arm],
                'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001,
                'gate_mode': model_for(arm).gate_mode, 'contrast_head': True,
                'control': model_for(arm).control, 'arm': 'multilevel'}
    if any(payload.get(k) != v for k, v in expected.items()):
        raise ValueError(f'{arm}: checkpoint identity differs')
    expected_keys = {(x['sequence'], x['frame']) for x in manifest['samples'] if x['split']=='validation'}
    rows = frozen['rows']; mapping = {(x['sequence'], x['frame']): x for x in rows}
    if len(rows) != 50 or len(mapping) != 50 or set(mapping) != expected_keys:
        raise ValueError(f'{arm}: validation prediction coverage differs')
    for row in rows:
        for name in ('baseline', 'refined'):
            path = Path(row[name]).resolve()
            if not path.is_relative_to(folder.resolve()) or digest(path) != row[name+'_sha256']:
                raise ValueError(f'{arm}: frozen prediction changed')
    return {'folder': folder, 'completion': completion, 'payload': payload, 'mapping': mapping}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('manifest', 'plan', 'parity', 'run-root', 'output'):
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args(); guard = install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest, manifest_sha = load_prepared(a.manifest); plan = require_plan(a.plan, manifest, manifest_sha)
    require_parity(a.parity, plan, a.plan, manifest_sha)
    if environment_identity() != plan['environment']: raise ValueError('Replay environment differs')
    runs = {arm: check_frozen_arm(a.run_root/arm, arm, manifest, manifest_sha, a.plan, a.parity, plan) for arm in ARMS}
    targets = sorted((x for x in manifest['samples'] if x['split']=='validation'), key=lambda x: (x['sequence'], x['frame']))
    baseline = {}
    # All four runs are complete and hashed before the first replay; no labels
    # are opened in this process, including after the replays complete.
    for arm, run in runs.items():
        model = model_for(arm)
        model.load_trained_state(run['payload'], backbone_sha256=manifest['checkpoint_sha256'],
            feature_pipeline_sha256=manifest['feature_pipeline_sha256'], train_manifest_sha256=manifest_sha)
        model.to('mps').eval()
        with torch.no_grad():
            for item in targets:
                early, deep, logits, shape = read_capture(item); args = (early.to('mps'), deep.to('mps'), logits.to('mps'))
                predicted = {'baseline': (to_video(args[2], shape)>0).cpu().numpy()[0,0],
                             'refined': (to_video(model(*args), shape)>0).cpu().numpy()[0,0]}
                row = run['mapping'][(item['sequence'], item['frame'])]
                if list(shape) != row['shape']: raise ValueError('Frozen prediction geometry differs')
                for name, actual in predicted.items():
                    with Image.open(row[name]) as img: saved = np.asarray(img.convert('L'))>0
                    if not np.array_equal(saved, actual): raise ValueError(f'{arm}: frozen mask replay differs')
                key = (item['sequence'], item['frame'])
                if key in baseline and baseline[key] != row['baseline_sha256']:
                    raise ValueError('Native baseline differs across arms')
                baseline[key] = row['baseline_sha256']
        del model
    if guard['open_attempts'] != 0: raise ValueError('Mask-label open attempted during replay')
    result = {'passed': True, 'scope': 'All frozen validation masks replayed; no label mask opened.',
              'manifest_sha256': manifest_sha, 'plan_sha256': digest(a.plan), 'parity_sha256': digest(a.parity),
              'dependency_sha256': dependencies(), 'environment': environment_identity(), 'label_guard': dict(guard),
              'runs': {arm: {'completion_path': str((r['folder']/'completion.json').resolve()),
                              'completion_sha256': digest(r['folder']/'completion.json'),
                              'checkpoint_sha256': digest(r['folder']/'checkpoint.pt'),
                              'frozen_sha256': digest(r['folder']/'frozen_predictions.json')} for arm, r in runs.items()},
              'runner_sha256': digest(Path(__file__))}
    out = validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    file = out/'verification.json'; file.write_text(json.dumps(result, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'verification_sha256': digest(file)}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'arms': len(runs), 'validation_frames': len(targets), 'label_open_attempts': 0}))


if __name__ == '__main__':
    main()
