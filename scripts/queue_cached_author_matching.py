"""Sequential cache gate, broader fixed-history experiment and frozen analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_regional_refinement import write
from scripts.summarize_cached_author_matching import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        plan = json.loads(args.plan.read_text())
        protocol = plan['protocol']
        runner_name = protocol.get('runner', 'scripts/run_cached_author_matching_diagnostic.py')
        if runner_name not in plan['source_hashes']:
            raise RuntimeError('Diagnostic runner must be included in frozen source hashes')
        analysis_sources = {name: digest(ROOT / name) for name in
                            ('scripts/summarize_cached_author_matching.py', 'evaluation/statistics.py')}
        write(out / 'analysis_sources.json', analysis_sources)
        write(out / 'queue_setup.json', {'plan': str(args.plan), 'sha256': digest(args.plan),
              'queue_source_sha256': digest(Path(__file__)), 'started': time.time()})
        videos = {**load_manifest(ROOT / protocol['manifest'], 'dev'),
                  **load_manifest(ROOT / protocol['manifest'], 'test')}
        for sequence, specification in plan['sequences'].items():
            for index, sha in specification['annotation_sha256'].items():
                if digest(videos[sequence][int(index)].mask_path) != sha:
                    raise RuntimeError('Evaluation annotations changed after plan freezing')
        for phase in ('cache_gate', 'broader'):
            command = [sys.executable, '-u', str(ROOT / runner_name),
                       '--plan', str(args.plan), '--output-dir', str(out / phase)]
            if phase == 'cache_gate':
                command.append('--cache-gate')
            else:
                command.extend(['--gate', str(out / 'cache_gate')])
            with (out / f'{phase}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                write(out / 'progress.json', {'phase': phase, 'pid': child.pid, 'started': time.time(), 'command': command})
                if child.wait() != 0:
                    raise RuntimeError(f'{phase} failed; see {phase}.log')
        if any(digest(ROOT / name) != sha for name, sha in analysis_sources.items()):
            raise RuntimeError('Predeclared analysis implementation changed during the experiment')
        report = summarize(out / 'broader', args.plan)
        write(out / 'completion.json', {'complete': True, 'scope': 'broader causal diagnostic; no full propagation',
              'report': str(report), 'summary_sha256': digest(out / 'broader/summary.json')})
        write(out / 'progress.json', {'phase': 'completed', 'time': time.time()})
    except Exception as exc:
        write(out / 'failure.json', {'error': str(exc), 'type': type(exc).__name__})
        write(out / 'progress.json', {'phase': 'failed', 'error': str(exc), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
