"""Source-encoding part of the oracle gate; no oracle query scoring or training."""
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
from scripts.mavos_value_utils import activate_author_repo, atomic_json, prepare
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.prepare_mask_memory_headroom import DESIGN
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import (
    CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _guard_sources, _model_digest, _source_hashes, _tree_digest,
)


def run(out, plan_path):
    out = validate_output(out)
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((plan_path.parent / 'completion.json').read_text())
    if not receipt['passed'] or digest(plan_path) != receipt['cases_sha256']:
        raise RuntimeError('Case-plan receipt is incomplete or altered')
    _guard_sources(plan['identity']['source_sha256'])
    if digest(DESIGN) != plan['identity']['protocol_sha256']:
        raise RuntimeError('Frozen oracle protocol changed')
    design, protocol = json.loads(DESIGN.read_text()), json.loads(PROTOCOL.read_text())
    sources = _source_hashes()
    for path in ('scripts/verify_mask_memory_teacher.py', 'scripts/mask_memory_teacher.py',
                 'modeling/native_memory_layout.py', 'modeling/memory_value_oracle.py',
                 'modeling/mavos_focal_value_modulation.py', 'protocols/mask_memory_value_headroom_v1.json',
                 'tests/test_memory_value_oracle.py', 'tests/test_mavos_focal_value_modulation.py',
                 'tests/test_mask_memory_headroom_plan.py', 'scripts/prepare_mask_memory_headroom.py'):
        sources[path] = digest(ROOT / path)
    identity = {'case_plan_sha256': digest(plan_path), 'sources': sources,
                'checkpoint_sha256': digest(CHECKPOINT_DEFAULT), 'threads': 1,
                'torch_version': torch.__version__, 'torch_build': torch.__config__.show()}
    if identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
        raise RuntimeError('Native checkpoint changed')
    atomic_json(out / 'identity.json', identity)
    for path in sources:
        dest = out / 'source' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / path).read_bytes())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(design['seed'])
    repo, revision = activate_author_repo()
    if revision != protocol['model']['sam2_revision']:
        raise RuntimeError('Native SAM2 source changed')
    videos = load_manifest(MANIFEST, 'dev')
    results = {'scope': 'Native source encoder only; full oracle read gate is still pending',
               'identity': identity, 'sequences': {}, 'oracle_query_scoring_started': False,
               'full_headroom_diagnostic_authorized': False, 'efficacy_authorized': False}
    for name in design['engineering_gate']['videos']:
        if name not in protocol['data']['train_sequences']:
            raise RuntimeError('Only declared TRAIN fixtures may be opened')
        start = plan['sequences'][name]['prompt_manifest_offset']
        frames = videos[name][start:start + 9]
        if start != prompt_index(videos[name], {'initialization': 'first_labeled_frame'}) or len(frames) != 9:
            raise RuntimeError('Fixed gate prompt/frames changed')
        model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
            checkpoint=CHECKPOINT_DEFAULT, model_config=protocol['model']['configuration']), repo)
        model.requires_grad_(False).eval()
        core = _model_digest(model)
        with tempfile.TemporaryDirectory(dir=out) as temporary:
            state, prompt = prepare(model, frames, Path(temporary))
            if prompt != 0:
                raise RuntimeError('Gate prompt changed')
            captured_masks = {}
            with capture_native_mask_inputs(model) as capture, torch.inference_mode():
                iterator = model.propagate_in_video(state, start_frame_idx=0)
                try:
                    for frame, ids, _ in iterator:
                        if ids != [1]:
                            raise RuntimeError('Unexpected source fixture owner')
                        if capture['count'] != frame + 1:
                            raise RuntimeError('Native encoder call/frame association changed')
                        captured_masks[frame] = clone_tensors(capture['latest'])
                finally:
                    iterator.close()
            bank = state['output_dict_per_obj'][0]
            before = _tree_digest(bank)
            spans = selected_memory_frames(model, 8, bank, 9)
            if sum(s['stop'] - s['start'] for s in spans) != 28672:
                raise RuntimeError('Gate did not reach the native full capacity')
            rows = []
            for span in spans:
                source = span['frame']
                if source in bank['cond_frame_outputs']:
                    continue  # The native GT-conditioned prompt stays unchanged.
                stored = clone_tensors(bank['non_cond_frame_outputs'][source])
                features = capture_current_features(model, state, source)
                gt = torch.from_numpy((load_label_mask(frames[source].mask_path, [1]) == 1).copy())[None, None]
                teacher = same_frame_teachers(model, features, stored, gt, captured_masks[source])
                rows.append({'local_source_frame': source, 'manifest_source_frame': frames[source].frame_index,
                    'span': span, 'image_sha256': digest(frames[source].image_path),
                    'mask_sha256': digest(frames[source].mask_path),
                    'native_reencoding_exact': teacher['native_reencoding_exact'],
                    'native_presence_logit': teacher['native_presence_logit'],
                    'gt_presence_logit': teacher['gt_presence_logit'],
                    'native_value_sha256': _tree_digest(teacher['native_reencoded']),
                    'mask_only_sha256': _tree_digest(teacher['mask_only']),
                    'mask_and_presence_sha256': _tree_digest(teacher['mask_and_presence'])})
                _guard_sources(sources)
            if len(rows) != 6 or _tree_digest(bank) != before:
                raise RuntimeError('Unexpected source coverage or native-bank mutation')
            model.reset_state(state)
        if _model_digest(model) != core:
            raise RuntimeError('Native SAM2 parameters changed during teacher encoding')
        results['sequences'][name] = rows
        atomic_json(out / 'progress.json', results)
        print(json.dumps({'source_encoding_checked': name, 'sources': len(rows)}), flush=True)
        del model
    _guard_sources(sources)
    atomic_json(out / 'results.json', results)
    atomic_json(out / 'completion.json', {'passed': True, 'results_sha256': digest(out / 'results.json'),
        'scope': results['scope'], 'full_headroom_diagnostic_authorized': False, 'efficacy_authorized': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--case-plan', required=True, type=Path)
    args = parser.parse_args()
    existed = args.output_dir.exists()
    try:
        run(args.output_dir, args.case_plan)
    except Exception as error:
        if not existed and args.output_dir.exists():
            atomic_json(args.output_dir / 'failure.json', {'type': type(error).__name__, 'error': str(error)})
        raise
