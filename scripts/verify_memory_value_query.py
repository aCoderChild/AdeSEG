"""Full-capacity TRAIN-only oracle-read integrity gate; no efficacy scoring."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.complementary_memory.training import capture_current_features, clone_tensors
from modeling.native_memory_layout import selected_memory_frames
from scripts.mask_memory_teacher import capture_native_mask_inputs, same_frame_teachers
from scripts.memory_value_query import MEMORY_FIELDS, query, spatial_values
from scripts.mavos_value_utils import activate_author_repo, atomic_json, prepare
from scripts.memory_token_io import digest, validate_output
from scripts.prepare_mask_memory_headroom import DESIGN
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _guard_sources, _model_digest, _tree_digest,
)


def run(out, encoder_gate, plan_path):
    # Reject stale/altered encoder receipts before native or oracle inference.
    completion = json.loads((encoder_gate / 'completion.json').read_text())
    prior = json.loads((encoder_gate / 'results.json').read_text())
    if not completion['passed'] or digest(encoder_gate / 'results.json') != completion['results_sha256']:
        raise RuntimeError('Encoder gate is incomplete or altered')
    _guard_sources(prior['identity']['sources'])
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != prior['identity']['case_plan_sha256']:
        raise RuntimeError('Encoder receipt belongs to another case plan')
    design, protocol = json.loads(DESIGN.read_text()), json.loads(PROTOCOL.read_text())
    if digest(DESIGN) != plan['identity']['protocol_sha256']:
        raise RuntimeError('Oracle protocol changed')
    sources = {**prior['identity']['sources']}
    for name in ('scripts/memory_value_query.py', 'scripts/verify_memory_value_query.py',
                 'modeling/complementary_memory/integration.py', 'tests/test_memory_value_query.py'):
        sources[name] = digest(ROOT / name)
    out = validate_output(out)
    out.mkdir(parents=True, exist_ok=False)
    identity = {'sources': sources, 'encoder_results_sha256': digest(encoder_gate / 'results.json'),
        'case_plan_sha256': digest(plan_path), 'checkpoint_sha256': digest(CHECKPOINT_DEFAULT),
        'threads': 1, 'torch_version': torch.__version__, 'torch_build': torch.__config__.show()}
    atomic_json(out / 'identity.json', identity)
    for relative in sources:
        path = out / 'source' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    atomic_json(out / 'protocol_frozen.json', design)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(design['seed'])
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision'] or identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Native source/checkpoint changed')
    videos = load_manifest(MANIFEST, 'dev')
    results = {'identity': identity, 'sequences': {}, 'scope': 'TRAIN oracle-read engineering gate',
               'efficacy_authorized': False, 'query_dice_scored': False}
    for name in design['engineering_gate']['videos']:
        if name not in design['data']['sequences'] or name not in protocol['data']['train_sequences']:
            raise RuntimeError('Only TRAIN engineering fixtures may be opened')
        start = plan['sequences'][name]['prompt_manifest_offset']
        frames = videos[name][start:start + 9]
        model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
            checkpoint=CHECKPOINT_DEFAULT, model_config=protocol['model']['configuration']), repo)
        model.requires_grad_(False).eval()
        core = _model_digest(model)
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            state, prompt = prepare(model, frames, Path(temporary))
            if prompt != 0 or len(frames) != 9:
                raise RuntimeError('Fixed engineering clip changed')
            masks, bank = {}, None
            with capture_native_mask_inputs(model) as capture, torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=0)
                try:
                    for frame, ids, _ in iterator:
                        if ids != [1] or capture['count'] != frame + 1:
                            raise RuntimeError('Native capture/owner association changed')
                        masks[frame] = clone_tensors(capture['latest'])
                        current = state['output_dict_per_obj'][0]
                        if frame == 7:
                            bank = clone_tensors(current)
                        if frame == 8:
                            target = clone_tensors({f: current['non_cond_frame_outputs'][8][f] for f in MEMORY_FIELDS})
                finally:
                    iterator.close()
            if bank is None or 8 in bank['non_cond_frame_outputs']:
                raise RuntimeError('Native query leaked into its own bank')
            frozen_bank = _tree_digest(bank)
            features = capture_current_features(model, state, 8)
            spans = selected_memory_frames(model, 8, bank, 9)
            expected = spatial_values(bank, spans)
            if expected.shape != (28672, 1, 64):
                raise RuntimeError('Gate did not reach seven native spatial frames')
            baseline = query(model, features, bank, 8, 9)
            if baseline['output_sha256'] != _tree_digest(target):
                raise RuntimeError('Cached query differs from actual native propagation')
            teachers = {}
            for span in spans:
                source = span['frame']
                if source in bank['cond_frame_outputs']:
                    continue
                visual = capture_current_features(model, state, source)
                gt = torch.from_numpy((load_label_mask(frames[source].mask_path, [1]) == 1).copy())[None, None]
                teachers[source] = same_frame_teachers(model, visual,
                    bank['non_cond_frame_outputs'][source], gt, masks[source])
            arms = {}
            for arm in ('identity_values', 'mask_only', 'mask_and_presence'):
                replacement = spatial_values(bank, spans, teachers, arm)
                result = query(model, features, bank, 8, 9, expected, replacement)
                if result['keys'] != baseline['keys'] or result['calls'] != baseline['calls']:
                    raise RuntimeError('Oracle changed native keys, positions, spans or pointer inputs')
                if arm == 'identity_values' and result['output_sha256'] != baseline['output_sha256']:
                    raise RuntimeError('Identity value wrapper differs from native')
                arms[arm] = {k: v for k, v in result.items() if k != 'output'}
            after = query(model, features, bank, 8, 9)
            if after['output_sha256'] != baseline['output_sha256'] or _tree_digest(bank) != frozen_bank:
                raise RuntimeError('Oracle changed subsequent native replay or the frozen bank')
            results['sequences'][name] = {'persistent_fixture_object_id': 1,
                'native_propagation_matches_cached_query': True, 'native_after_oracle_exact': True,
                'source_encoder_exact_count': len(teachers), 'spans': spans,
                'native': {k: v for k, v in baseline.items() if k != 'output'}, 'arms': arms,
                'bank_sha256': frozen_bank,
                'input_files': [{'image': digest(f.image_path), 'mask': digest(f.mask_path)} for f in frames]}
            model.reset_state(state)
        if _model_digest(model) != core:
            raise RuntimeError('Pretrained SAM2 parameters changed')
        _guard_sources(sources)
        atomic_json(out / 'progress.json', results)
        print(json.dumps({'read_gate_checked': name, 'spatial_tokens': 28672, 'layers': 4}), flush=True)
        del model
    atomic_json(out / 'results.json', results)
    atomic_json(out / 'completion.json', {'passed': True, 'results_sha256': digest(out / 'results.json'),
        'scope': results['scope'], 'full_train_diagnostic_integrity_passed': True,
        'efficacy_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--encoder-gate', required=True, type=Path)
    parser.add_argument('--case-plan', required=True, type=Path)
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        run(args.output_dir, args.encoder_gate, args.case_plan)
    except Exception as error:
        if not existed and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
