"""Inspect a live queued diagnostic without loading its large result arrays."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import validate_output


def process(pid, expected_script):
    result = subprocess.run(['ps', '-p', str(int(pid)), '-o', 'pid=,etime=,%cpu=,rss=,command='],
                            text=True, capture_output=True)
    row = result.stdout.strip()
    if not row:
        return {'pid': pid, 'live': False}
    values = row.split(maxsplit=4)
    if len(values) != 5 or expected_script not in values[4]:
        return {'pid': pid, 'live': False, 'pid_identity_mismatch': True, 'observed': row}
    return {'pid': int(values[0]), 'live': True, 'elapsed': values[1],
            'cpu_percent': float(values[2]), 'rss_mib': round(int(values[3])/1024, 1)}


def observe(folder):
    folder = validate_output(folder)
    setup = json.loads((folder / 'queue_setup.json').read_text())
    plan = json.loads(Path(setup['plan']).read_text())
    # A progress file may briefly be in the middle of a write. Such an
    # observation is inconclusive, never evidence for restarting a process.
    progress = json.loads((folder / 'progress.json').read_text())
    result = {'observed_utc_epoch': time.time(), 'folder': str(folder),
              'phase': progress['phase'], 'planned_targets': plan['counts']['targets']}
    for terminal in ('failure', 'completion'):
        path = folder / f'{terminal}.json'
        if path.exists():
            result['state'] = 'failed' if terminal == 'failure' else 'completed'
            result['terminal_receipt'] = json.loads(path.read_text())
            return result
    if 'pid' not in progress:
        result['state'] = 'no_child_handle'
        return result
    expected_runner = plan['protocol'].get('runner', 'scripts/run_cached_author_matching_diagnostic.py')
    child = process(progress['pid'], str(ROOT / expected_runner))
    result['child'] = child
    phase_dir = folder / progress['phase']
    detail_path = phase_dir / 'progress.json'
    if detail_path.exists():
        detail = json.loads(detail_path.read_text())
        result['progress'] = detail
        result['progress_file_age_seconds'] = round(time.time()-detail_path.stat().st_mtime, 1)
        if progress['phase'] == 'broader' and 'sequence' in detail:
            total = 0
            for sequence in plan['protocol']['sequences']:
                if sequence == detail['sequence']:
                    total += detail['completed_in_sequence']
                    break
                total += len(plan['sequences'][sequence]['schedule']['targets'])
            result['completed_targets_lower_bound'] = total
    failure = phase_dir / 'failure.json'
    if failure.exists():
        result['state'] = 'child_failed'
        result['terminal_receipt'] = json.loads(failure.read_text())
    else:
        result['state'] = 'running' if child['live'] else 'child_handle_missing_check_queue'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--snapshot', type=Path)
    args = parser.parse_args()
    try:
        observation = observe(args.folder)
    except json.JSONDecodeError:
        observation = {'state': 'observation_inconclusive_retry',
                       'note': 'A state file is being rewritten; do not restart the run.'}
    if args.snapshot:
        validate_output(args.snapshot).write_text(json.dumps(observation, indent=2)+'\n')
    print(json.dumps(observation, indent=2))
