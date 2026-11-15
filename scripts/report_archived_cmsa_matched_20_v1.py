"""Compare independently audited historical CMSA with audited direct-prompt results.

This uses stored scores only. It does not open images, predictions, or labels.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
CMSA = OUT / 'cmsa_net_polypgen_v1'
CONTRAST = OUT / 'contrast_refinement_v1'
PATHS = {
    'cmsa': CMSA / 'automatic_target_cohort_v3_scores_attempt01/results.json',
    'cmsa_audit': CMSA / 'automatic_target_cohort_v3_independent_audit_attempt01/audit.json',
    'verification': CMSA / 'automatic_target_cohort_v3_verification_attempt01/verification.json',
    'merged': CONTRAST / 'all_23_direct_prompt_comparison_v1_attempt01/results.json',
    'additional': CONTRAST / 'additional_author_comparator_v2_01_uxztj/scores/results.json',
    'priority': OUT / 'sam2long_automatic_priority_v1/four_system_report_attempt01/results.json',
    'factorial': CONTRAST / 'boundary_video_factorial_v1_7c3zaram/scores/results.json',
    'reconciliation': CONTRAST / 'additional_author_comparator_v2_01_uxztj_independent_audit_v3_attempt01/reconciliation.json',
}
METHODS = ('native', 'samurai', 'sam2long', 'lean', 'refiner', 'cmsa')
SEQUENCES = tuple(f'seq{i}' for i in range(2, 23) if i != 7)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(data, hashes):
    c, a, v, m = (data[k] for k in ('cmsa', 'cmsa_audit', 'verification', 'merged'))
    require(all(x.get('passed') is True for x in (c, a, v, m, data['additional'], data['priority'], data['factorial'], data['reconciliation'])), 'Input proof not passed')
    require(a['error_count'] == 0 and a['score_sha256'] == hashes['cmsa'] and
            a['verification_sha256'] == c['verification_sha256'] == hashes['verification'], 'CMSA proof mismatch')
    require(v['reconciliation_sha256'] == m['inputs_sha256']['additional_reconciliation'] == hashes['reconciliation'] and
            v['prior_sha256'] == m['inputs_sha256']['priority_factorial_score'] == hashes['factorial'] and
            m['inputs_sha256']['additional_score'] == hashes['additional'] and
            m['inputs_sha256']['priority_four_score'] == hashes['priority'], 'Upstream score identity mismatch')
    require(set(c['rows']) == set(c['summaries']) == set(SEQUENCES), 'CMSA scope mismatch')
    table = {}
    for seq in SEQUENCES:
        source = data['priority']['rows'][seq] if seq in ('seq11', 'seq20') else data['additional']['rows'][seq]
        cmsa = c['rows'][seq]
        require(len(source) == len(cmsa), f'{seq}: frame count mismatch')
        for left, right in zip(source, cmsa, strict=True):
            for field in ('frame', 'source_frame_index', 'post_prompt', 'initialization_frame'):
                require(left[field] == right[field], f'{seq}: frame identity mismatch: {field}')
            require(left['native']['present'] == right['cmsa']['present'], f'{seq}: label presence mismatch')
        table[seq] = dict(m['table'][seq]); table[seq]['cmsa'] = c['summaries'][seq]
        for method, entry in table[seq].items():
            if entry is not None:
                require(entry['present_frames'] == c['summaries'][seq]['present_frames'] and
                        entry['empty_frames'] == c['summaries'][seq]['empty_frames'], f'{seq}: {method} denominator mismatch')
                if method == 'cmsa':
                    rows, key = cmsa, 'cmsa'
                elif seq in ('seq11', 'seq20') and method == 'refiner':
                    rows, key = data['factorial']['rows'][seq], 'D'
                    require(len(rows) == len(source), f'{seq}: factorial frame count mismatch')
                    for x, y in zip(rows, source, strict=True):
                        require(all(x[f] == y[f] for f in ('frame', 'source_frame_index', 'post_prompt', 'initialization_frame')) and
                                x['D']['present'] == y['native']['present'], f'{seq}: factorial frame mismatch')
                else:
                    rows = source
                    key = {'lean': 'A', 'refiner': 'D'}.get(method, method) if seq not in ('seq11', 'seq20') else method
                post = [r[key] for r in rows if r['post_prompt']]
                present = [r['dice'] for r in post if r['present']]
                empty = [r for r in post if not r['present']]
                require(len(present) == entry['present_frames'] and len(empty) == entry['empty_frames'] and
                        sum(bool(r['false_alarm']) for r in empty) == entry['false_alarms'], f'{seq}: {method} score count mismatch')
                if present:
                    require(math.isclose(sum(present)/len(present), entry['mean_present_dice'], rel_tol=1e-12, abs_tol=1e-12),
                            f'{seq}: {method} score mean mismatch')
    def aggregate(methods):
        seqs = [s for s in SEQUENCES if all(table[s].get(k) is not None and table[s][k]['present_frames'] for k in methods)]
        require(bool(seqs), 'No eligible patients for cohort')
        result = {}
        for method in methods:
            rows = [table[s][method] for s in seqs]
            n = sum(r['present_frames'] for r in rows)
            result[method] = {'patient_macro_dice': sum(r['mean_present_dice'] for r in rows)/len(rows),
                              'frame_weighted_dice': sum(r['mean_present_dice']*r['present_frames'] for r in rows)/n,
                              'present_frames': n, 'false_alarms': sum(r['false_alarms'] for r in rows),
                              'empty_frames': sum(r['empty_frames'] for r in rows)}
        return {'sequences': seqs, 'metrics': result}
    return {'table': table, 'all_20': aggregate(('native', 'samurai', 'sam2long', 'cmsa')),
            'all_six': aggregate(METHODS)}


def main():
    require(len(sys.argv) == 2, 'Provide a new report directory under outputs/experiment')
    output = Path(sys.argv[1]).resolve()
    require(output.is_relative_to(OUT.resolve()) and not output.exists(), 'Invalid or existing output')
    raw = {k: p.read_bytes() for k, p in PATHS.items()}
    hashes = {k: digest(v) for k, v in raw.items()}
    data = {k: json.loads(v) for k, v in raw.items()}
    result = compare(data, hashes)
    require(all(digest(p.read_bytes()) == hashes[k] for k, p in PATHS.items()), 'Source changed during comparison')
    result['input_sha256'] = hashes
    result['script_sha256'] = digest(Path(__file__).read_bytes())
    result['scope'] = ('Previously inspected sequences; CMSA historical CPU unprompted clips, SAM2Long offline. '
        'Historical CMSA lacks inference-time source-image hashes and a cryptographic setup-to-output link; '
        'exact common target frame identity and label presence are checked. Project Lean is modified; '
        'the six-method cohort is smaller. Seq23 is excluded for every method.')
    lines = ['# Historical CMSA: matched 20-sequence target-frame comparison', '', result['scope'], '']
    for key in ('all_20', 'all_six'):
        group = result[key]; lines += [f"## {key}: {len(group['sequences'])} patients", '', ', '.join(group['sequences']), '',
            '| Method | Patient macro Dice | Frame weighted Dice | Present frames | Empty false alarms |',
            '|---|---:|---:|---:|---:|']
        for name, row in group['metrics'].items():
            lines.append(f"| {name} | {row['patient_macro_dice']:.4f} | {row['frame_weighted_dice']:.4f} | {row['present_frames']} | {row['false_alarms']} / {row['empty_frames']} |")
        lines.append('')
    report = '\n'.join(lines) + '\n'; result['report_sha256'] = digest(report.encode())
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.md').write_text(report)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(report)


if __name__ == '__main__':
    main()
