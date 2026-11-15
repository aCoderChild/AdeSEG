"""Sequential gate, exact replay comparison, then wider diagnostics on Drive."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output
from scripts.run_regional_refinement import write
from scripts.summarize_token_identification_diagnostic import summarize


def compare(reference, actual):
    compared = 0
    for folder in [reference, actual]:
        done = json.loads((folder / 'completion.json').read_text())
        if not done.get('engineering_passed') or digest(folder / 'results.json') != done['results_sha256']:
            raise RuntimeError('Missing or changed completion receipt')
    for sequence in ('seq2', 'seq3'):
        files = sorted((reference / sequence).glob('*.npz'))
        if not files or {p.name for p in files} != {p.name for p in (actual / sequence).glob('*.npz')}:
            raise RuntimeError('Replay artifact coverage differs')
        for path in files:
            with np.load(path) as first, np.load(actual / sequence / path.name) as second:
                if set(first.files) != set(second.files):
                    raise RuntimeError('Replay array names differ')
                for name in first.files:
                    if not np.array_equal(first[name], second[name]):
                        raise RuntimeError(f'Replay mismatch: {sequence}/{path.name}/{name}')
                    compared += 1
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        for phase, protocol in [('gate', 'token_identification_diagnostic_v1.json'),
                                ('wider', 'token_identification_diagnostic_wider_v2.json')]:
            command = [sys.executable, '-u', str(ROOT / 'scripts/run_token_identification_diagnostic.py'),
                       '--protocol', str(ROOT / 'protocols' / protocol), '--output-dir', str(out / phase)]
            with (out / f'{phase}.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                write(out / 'progress.json', {'phase': phase, 'pid': child.pid, 'started': time.time(), 'command': command})
                if child.wait() != 0:
                    raise RuntimeError(f'{phase} failed; see {phase}.log')
            summarize(out / phase)
            if phase == 'gate':
                arrays = compare(args.reference, out / phase)
                write(out / 'gate_replay.json', {'passed': True, 'bit_identical_arrays': arrays,
                                               'reference': str(args.reference)})
            else:
                gate_sources = json.loads((out / 'gate/setup.json').read_text())['sources']
                wider_sources = json.loads((out / 'wider/setup.json').read_text())['sources']
                if gate_sources != wider_sources:
                    raise RuntimeError('Source changed between engineering gate and wider diagnostics')
        write(out / 'completion.json', {'complete': True, 'efficacy_run': False,
              'report': str(out / 'wider/report.md'), 'note': 'Review wider diagnostics before selecting a new ToMe mechanism.'})
        write(out / 'progress.json', {'phase': 'completed', 'time': time.time()})
    except Exception as exc:
        write(out / 'failure.json', {'error': str(exc), 'time': time.time()})
        write(out / 'progress.json', {'phase': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
