"""Isolated instrumented-cost worker; reuses the verified value evaluator.

Run one fresh process per scheduled job after training and validation finish.
This measures evaluation throughput, including metrics and mask I/O, not pure
network latency. Real-frame benchmark verification is required before release.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.mavos_value_utils import atomic_json

CONDITIONS = ('native', 'segmentation_only', 'segmentation_plus_consistency')


def schedule():
    jobs = []
    for sequence in ('seq2', 'seq3'):
        jobs.extend({'sequence': sequence, 'phase': 'warmup', 'condition': c}
                    for c in CONDITIONS)
        for repeat in range(3):
            order = CONDITIONS[repeat:] + CONDITIONS[:repeat]
            jobs.extend({'sequence': sequence, 'phase': f'repeat{repeat + 1}',
                         'condition': c} for c in order)
    return jobs


def prediction_identity(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    identity = [(r['frame'], r['prediction_sha256'], r['mask_sha256']) for r in rows]
    if not identity or len({r[0] for r in identity}) != len(identity):
        raise RuntimeError('Missing or duplicate benchmark prediction frames')
    return identity


def run(args):
    # Heavy model imports occur only in the isolated worker, never the planner.
    import torch
    from datasets.video import load_manifest
    from scripts.mavos_value_utils import activate_author_repo
    from scripts.rde_value_evaluation import evaluate_condition, load_adapter
    from scripts.run_author_baseline import build_author_predictor
    from scripts.run_rde_value_training import verify_receipt
    from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT, MANIFEST, _guard_sources

    design = json.loads((ROOT / 'protocols/rde_value_repair_efficacy_polypgen_v1.json').read_text())
    training, training_sha = verify_receipt(args.training_root)
    validation, validation_sha = verify_receipt(args.validation_root)
    gate, gate_sha = verify_receipt(args.benchmark_gate)
    if not gate.get('benchmark_worker_real_frame_parity_passed') or gate.get('worker_sha256') != digest(Path(__file__)):
        raise RuntimeError('Benchmark-specific real-frame gate is not available')
    if training.get('engineering') is not False or not validation.get('technical_evaluation_passed'):
        raise RuntimeError('Fixed training and validation must complete before benchmarking')
    identity = json.loads((args.training_root / 'identity.json').read_text())
    _guard_sources(identity['sources'])
    evaluation_identity = json.loads((args.validation_root / 'identity.json').read_text())
    _guard_sources(evaluation_identity['sources'])
    if (evaluation_identity['training_results_sha256'] != training_sha or
            evaluation_identity['training_identity_sha256'] != digest(args.training_root / 'identity.json')):
        raise RuntimeError('Validation belongs to a different trained study')
    if identity['epochs'] != 3 or identity['seed'] != design['training']['primary_seed']:
        raise RuntimeError('Only the fixed primary final epoch is accepted')
    if (digest(CHECKPOINT_DEFAULT) != design['model']['checkpoint_sha256'] or
            digest(MANIFEST) != design['data']['manifest_sha256'] or
            torch.__version__ != design['training']['torch_version']):
        raise RuntimeError('Frozen model/data/runtime differs')
    out = validate_output(args.output_dir)
    # Deliberately refuse cached timings and preserve any failed attempt.
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    repo, revision = activate_author_repo()
    if revision != design['model']['sam2_revision']:
        raise RuntimeError('SAM2 revision differs')
    model = build_author_predictor(argparse.Namespace(method='sam2', device='cpu',
        checkpoint=CHECKPOINT_DEFAULT, model_config=design['model']['configuration']), repo)
    model.requires_grad_(False).eval()
    adapter = None
    if args.condition != 'native':
        path = args.training_root / 'training' / args.condition / 'adapter_final.pt'
        receipt, _ = verify_receipt(path.parent)
        if receipt != training['arms'][args.condition] or digest(path) != receipt['final_adapter_file_sha256']:
            raise RuntimeError('Final artifact/receipt differs')
        adapter = load_adapter(path, digest(args.training_root / 'identity.json'), args.condition)
    provenance = {'training_results_sha256': training_sha, 'validation_results_sha256': validation_sha,
        'benchmark_gate_sha256': gate_sha, 'worker_sha256': digest(Path(__file__)),
        'condition': args.condition, 'sequence': args.sequence, 'phase': args.phase,
        'platform': platform.platform(), 'machine': platform.machine(), 'torch_version': torch.__version__,
        'threads': torch.get_num_threads(), 'total_model_parameters': sum(p.numel() for p in model.parameters()),
        'adapter_parameters': 0 if adapter is None else sum(p.numel() for p in adapter.parameters())}
    atomic_json(out / 'identity.json', provenance)
    started = time.monotonic()
    summary = evaluate_condition(model, load_manifest(MANIFEST, 'dev')[args.sequence], adapter,
                                 args.sequence, out, digest(out / 'identity.json'))
    elapsed = time.monotonic() - started
    predictions = prediction_identity(out / 'evaluation' / args.sequence / 'frames.jsonl')
    if args.phase != 'warmup':
        if args.warmup is None or predictions != prediction_identity(args.warmup / 'evaluation' / args.sequence / 'frames.jsonl'):
            raise RuntimeError('Repeated predictions differ from warmup; no cost comparison released')
        warm = json.loads((args.warmup / 'identity.json').read_text())
        for key in provenance.keys() - {'phase'}:
            if warm[key] != provenance[key]:
                raise RuntimeError(f'Warmup ownership/runtime differs: {key}')
    _guard_sources(identity['sources'])
    _guard_sources(evaluation_identity['sources'])
    atomic_json(out / 'cost.json', {'instrumented_evaluator_seconds': elapsed,
        'scope': 'evaluate_condition wall time, including hashes, metrics, mask I/O and cache verification; excludes model construction',
        'target_frames': summary['target_frames'], 'instrumented_fps': summary['target_frames'] / elapsed,
        'process_peak_rss_bytes': summary['process_peak_rss_bytes'], 'peak_rss_scope': summary['peak_rss_scope'],
        'prediction_repeatability_passed': args.phase != 'warmup'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-schedule', action='store_true')
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--sequence', choices=('seq2', 'seq3'))
    parser.add_argument('--phase', choices=('warmup', 'repeat1', 'repeat2', 'repeat3'))
    for name in ('training-root', 'validation-root', 'benchmark-gate', 'output-dir', 'warmup'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.print_schedule:
        print(json.dumps(schedule(), indent=2))
    else:
        if any(getattr(args, k) is None for k in ('condition', 'sequence', 'phase', 'training_root', 'validation_root', 'benchmark_gate', 'output_dir')):
            parser.error('A worker requires condition, sequence, phase and all prerequisite/output paths')
        run(args)
