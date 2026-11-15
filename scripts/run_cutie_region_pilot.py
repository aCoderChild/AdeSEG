"""Exploratory Cutie-summary gate pilot using the verified CFBI workflow."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.cutie_memory import CutieRegionGate, CUTIE_REVISION
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, prompt_index, validate_output
from scripts.run_regional_refinement import write
import scripts.run_cfbi_region_pilot as pilot

SOURCES = ['protocols/cutie_region_gate_pilot.json', 'scripts/run_cutie_region_pilot.py',
           'scripts/run_cfbi_region_pilot.py', 'modeling/cutie_memory.py', 'modeling/cfbi_region_gate.py',
           'modeling/cfbi_training.py', 'modeling/region_definitions.py', 'modeling/region_tokens.py',
           'tests/test_cutie_memory.py', 'external/Cutie/cutie/model/transformer/object_summarizer.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'pilot-dir', 'source', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--presence-capture', type=Path)
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        args.method, args.device = 'sam2', 'cpu'
        root = args.pilot_dir.parent
        for file in ('cutie_real_multimemory_gate.json', 'cutie_zero_full_seq2_v1/completion.json', 'cutie_nonzero_seq3_v1/completion.json'):
            if not json.loads((root / file).read_text())['passed']:
                raise RuntimeError(f'Engineering gate failed: {file}')
        source_hashes = {path: digest(ROOT / path) for path in SOURCES}
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
        if source_hashes != {path: digest(ROOT / path) for path in SOURCES}:
            raise RuntimeError('Source changed during tests')
        previous = json.loads((args.pilot_dir / 'setup.json').read_text())
        if digest(args.checkpoint) != previous['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed')
        manifest = ROOT / 'data/polypgen_sequence.jsonl'
        if digest(manifest) != previous['manifest_sha256']:
            raise RuntimeError('Manifest changed')
        videos = {**load_manifest(manifest, 'dev'), **load_manifest(manifest, 'test')}
        protocol_path = 'protocols/cutie_presence_pilot.json' if args.presence_capture else SOURCES[0]
        protocol = json.loads((ROOT / protocol_path).read_text())
        source_hashes[protocol_path] = digest(ROOT / protocol_path)
        receipts = json.loads((args.source / 'inputs.json').read_text())
        names = protocol['training_sequences'] + protocol['validation_sequences']
        for name in names:
            for frame, receipt in zip(videos[name], receipts[name], strict=True):
                if frame.frame_index != receipt['frame'] or digest(frame.image_path) != receipt['image_sha256'] or digest(frame.mask_path) != receipt['mask_sha256']:
                    raise RuntimeError(f'Input changed: {name}/{frame.frame_index}')
        samples = json.loads((args.pilot_dir / 'training_samples.json').read_text())
        if len(samples) != 12 or any(digest(row['path']) != row['sha256'] for row in samples):
            raise RuntimeError('Training samples changed')
        if args.presence_capture:
            gradient_gate = json.loads((root / 'seq13_presence_replay_v2' / 'presence_gradient_gate.json').read_text())
            if not gradient_gate['passed'] or not gradient_gate['target_present']:
                raise RuntimeError('Real missed-object gradient gate failed')
            capture = json.loads((args.presence_capture / 'completion.json').read_text())
            capture_gate = json.loads((args.presence_capture / 'cutie_presence_gradient_gate.json').read_text())
            extra = json.loads((args.presence_capture / 'training_samples.json').read_text())
            if not capture['passed'] or [(r['sequence'], r['frame']) for r in extra] != [('seq4', 24), ('seq4', 25)]:
                raise RuntimeError('Missing verified development misses')
            if not capture_gate['passed'] or not capture_gate['target_present'] or capture_gate['object_score'] >= 0:
                raise RuntimeError('Cutie missed-object gradient gate failed')
            if any(digest(r['path']) != r['sha256'] for r in extra):
                raise RuntimeError('Development misses changed')
            samples += extra
            rng = np.random.default_rng(20261031)
            schedule = np.concatenate((rng.choice(12, size=18, replace=True), rng.choice([12, 13], size=6, replace=True)))
            rng.shuffle(schedule)
        else:
            schedule = np.asarray(json.loads((args.pilot_dir / 'schedule.json').read_text()))
            if schedule.tolist() != np.random.default_rng(20261031).choice(12, size=24, replace=True).tolist():
                raise RuntimeError('Training schedule changed')
        baseline = {}
        for name in protocol['validation_sequences']:
            path = args.source / 'cache' / name / 'native_frames.json'
            if digest(path) != previous['validation_native_frames_sha256'][name]:
                raise RuntimeError(f'Native baseline changed: {name}')
            rows = json.loads(path.read_text())
            prompt = prompt_index(videos[name], {'initialization': 'first_labeled_frame'})
            pilot.verify_baseline_coverage(rows, prompt, len(videos[name]))
            baseline[name] = {**propagation_summary(rows), **{k: float(np.mean([r['scores'][k] for r in rows if r['present']])) for k in ('iou', 'boundary_f')}}
        repo, revision = pilot.activate_author_repo()
        pilot.RegionGate = CutieRegionGate
        write(args.output_dir / 'setup.json', {'protocol': protocol, 'sources': source_hashes, 'checkpoint_sha256': previous['checkpoint_sha256'],
              'manifest_sha256': previous['manifest_sha256'], 'sam2_revision': revision, 'cutie_revision': CUTIE_REVISION,
              'patient_independence_verified': False, 'samples': samples, 'schedule': schedule.tolist()})
        for mode in ('regional', 'shuffled', 'generic'):
            pilot.train(args, samples, repo, mode, schedule, presence_weight=1.0 if args.presence_capture else 0.0)
        results = {'native': baseline, 'conditions': {}, 'scope': protocol['interpretation']}
        for mode in ('regional', 'shuffled', 'generic'):
            info = json.loads((args.output_dir / mode / 'training.json').read_text())
            results['conditions'][mode] = {}
            for name in protocol['validation_sequences']:
                rows, summary = pilot.validate(args, videos[name], repo, mode, name, args.output_dir / mode / 'head.pt', info['checkpoint_sha256'])
                write(args.output_dir / mode / f'{name}_frames.json', rows)
                results['conditions'][mode][name] = summary
        if source_hashes != {path: digest(ROOT / path) for path in source_hashes}:
            raise RuntimeError('Source changed during pilot')
        write(args.output_dir / 'results.json', results)
        (args.output_dir / 'results.md').write_text(pilot.report(results, protocol['validation_sequences']))
        write(args.output_dir / 'completion.json', {'complete': True, 'validation_sequences': protocol['validation_sequences']})
        write(args.output_dir / 'progress.json', {'phase': 'complete'})
    except BaseException as error:
        write(args.output_dir / 'failure.json', {'error': repr(error)})
        raise

if __name__ == '__main__':
    main()
