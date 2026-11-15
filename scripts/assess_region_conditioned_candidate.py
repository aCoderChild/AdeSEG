#!/usr/bin/env python3
"""Rescore the region-conditioned study for exploratory development gating.

This reuses the existing independent four-arm auditor and never authorizes a
live runtime by itself. The fixed-read validation set was previously inspected;
the conservative screen is post hoc and is not an independent or final test.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.assess_refiner_candidate_promotion import screen
from scripts.audit_region_conditioned_hardcase_results import run as replay_and_audit
from scripts.memory_token_io import digest, validate_output


def development_decision(audit):
    arms = audit['arms']
    keyed = {}
    for name in ('region', 'global', 'shuffled', 'mask_only'):
        rows = [x for x in arms[name]['rows'] if x['split'] == 'validation']
        mapping = {(x['sequence'], x['frame']): x for x in rows}
        if len(mapping) != len(rows):
            raise ValueError('Duplicate validation frame')
        keyed[name] = mapping
    keys = set(keyed['region'])
    if not keys or any(set(keyed[name]) != keys for name in keyed):
        raise ValueError('Matched region-control frames differ')
    metrics = []
    for seq, frame in sorted(keys):
        r = keyed['region'][(seq, frame)]
        if any(keyed[name][(seq, frame)]['present'] != r['present'] for name in keyed):
            raise ValueError('Matched presence labels differ')
        if any(keyed[name][(seq, frame)]['baseline'] != r['baseline'] for name in keyed):
            raise ValueError('Matched baseline masks differ')
        metrics.append({'sequence': seq, 'frame': frame, 'labeled_present': r['present'],
                        'baseline': r['baseline'], 'candidate': r['refined'],
                        'control': keyed['mask_only'][(seq, frame)]['refined']})
    result = screen(metrics)
    if 'checks' in result:
        present = [key for key in sorted(keys) if keyed['region'][key]['present']]
        videos = sorted({key[0] for key in present})
        extra = {}
        for name in ('global', 'shuffled'):
            mean = float(np.mean([np.mean([keyed[name][key]['refined']['dice'] for key in present if key[0] == video])
                                  for video in videos]))
            extra[name] = mean
            result['checks']['beats_' + name + '_present_dice'] = result['mean_per_video_present_dice']['candidate'] > mean
        result['additional_control_mean_per_video_present_dice'] = extra
        result['eligible'] = all(result['checks'].values())
    return result


def assess(manifest, candidate, global_control, shuffled_control, mask_control, output):
    out = validate_output(output)
    out.mkdir(parents=True, exist_ok=False)
    inputs = {'region': candidate, 'global': global_control,
              'shuffled': shuffled_control, 'mask_only': mask_control}
    replay_and_audit(manifest, inputs, out / 'replay')
    audit_path = out / 'replay/audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit.get('passed'):
        raise ValueError('Independent four-arm audit failed')
    result = {'audit_passed': True, 'decision': development_decision(audit),
              'manifest_sha256': digest(manifest / 'manifest.json'),
              'candidate_checkpoint_sha256': digest(candidate / 'checkpoint.pt'),
              'global_checkpoint_sha256': digest(global_control / 'checkpoint.pt'),
              'shuffled_checkpoint_sha256': digest(shuffled_control / 'checkpoint.pt'),
              'control_checkpoint_sha256': digest(mask_control / 'checkpoint.pt'),
              'replay_audit_sha256': digest(audit_path), 'auditor_sha256': digest(Path(__file__)),
              'scope': 'Post hoc, already-inspected fixed-read development data; not independent testing or live-video authorization.'}
    (out / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['decision'], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'candidate', 'global-control', 'shuffled-control', 'mask-control', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assess(args.manifest, args.candidate, args.global_control, args.shuffled_control, args.mask_control, args.output)


if __name__ == '__main__':
    main()
