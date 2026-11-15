"""Summarize completed bounded diagnostics without implying efficacy."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output


def summarize(folder):
    folder = validate_output(folder)
    completion = json.loads((folder / 'completion.json').read_text())
    if not completion.get('engineering_passed') or completion.get('efficacy_evaluated'):
        raise ValueError('Expected a completed engineering-only diagnostic')
    if digest(folder / 'results.json') != completion['results_sha256']:
        raise ValueError('Results changed after completion')
    records = json.loads((folder / 'results.json').read_text())
    lines = ['# Native memory region diagnostic', '',
             'Bounded seq2/seq3 fixtures. These are membership/attribution metrics, **not segmentation Dice gains**.', '',
             '| Sequence | Target | Source | TokenCut partition | Native replay |',
             '|---|---:|---:|---|---|']
    methods = set()
    for sequence, targets in records.items():
        for target in targets:
            for source in target['sources']:
                artifact = Path(source['artifact'])
                if digest(artifact) != source['sha256']:
                    raise ValueError(f'Artifact changed: {artifact}')
                status = source['partition_error'] or 'Computed; identity anchored to initial prompt'
                lines.append(f"| {sequence} | {target['frame']} | {source['source_frame']} | {status} | {'Pass' if target['native_replay_passed'] else 'Fail'} |")
                methods.update(source['metrics'])
    lines += ['', 'The table below excludes conditioning/prompt sources, because their masks were supplied during initialization. Only propagated source masks test identification beyond that prompt.',
              '', '| Signal | seq2 AUROC (propagated source) | seq3 AUROC (propagated source) |', '|---|---:|---:|']
    for method in sorted(methods):
        values = []
        for sequence in ('seq2', 'seq3'):
            aucs = [source['metrics'][method]['auroc'] for target in records[sequence]
                    for source in target['sources'] if source['temporal_slot'] != 0 and method in source['metrics']
                    and source['metrics'][method]['auroc'] is not None]
            values.append(f'{statistics.mean(aucs):.4f} (n={len(aucs)})' if aucs else 'N/A')
        lines.append(f"| {method} | {' | '.join(values)} |")
    lines += ['', 'AUROC is averaged across source–target reads within each sequence. Repeated source frames are correlated; these are descriptive diagnostics, not independent samples or significance tests.',
              '', 'Low gradients are not assigned background labels. Constant maps and single-class evaluation targets are explicitly recorded in results.json. A graph partition can be undefined even when native segmentation succeeds.',
              '', 'No full efficacy run is authorised by this engineering receipt alone. Remaining checks include fresh-process replay, wider fixed-frame coverage, temporal stability, multi-object isolation and matched-budget ToMe integration.', '']
    destination = folder / 'report.md'
    destination.write_text('\n'.join(lines))
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    print(summarize(parser.parse_args().folder))
