"""Audit source-bound oracle case coverage/invariants and recompute summaries.

This checks saved receipts and metric algebra; it does not independently rescore
unsaved masks. The separate fresh-process case replay supplies bounded runtime
reconstruction evidence. A partial audit can never release a completed study.
"""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.analyze_memory_value_headroom import ARMS,summarize
from scripts.mavos_value_utils import atomic_json
from scripts.memory_token_io import digest,validate_output
from scripts.run_token_conditioned_key_adapter import _guard_sources


def check_scores(row,present):
    scores=row['scores']
    if not math.isfinite(row['object_score']):
        raise RuntimeError('Saved object score is nonfinite')
    if any(not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):
        raise RuntimeError('Saved metric is nonfinite or outside [0,1]')
    if not math.isclose(scores['dice'],2*scores['iou']/(1+scores['iou']),abs_tol=1e-12,rel_tol=0):
        raise RuntimeError('Dice/IoU algebra differs')
    if row['object_score']<=0 and row['prediction_nonempty']:
        raise RuntimeError('Native absent score has a nonempty final mask')
    if not row['prediction_nonempty'] and scores['dice']!=(0 if present else 1):
        raise RuntimeError('Empty-mask Dice convention differs')
    if not present and scores['dice']!=(0 if row['prediction_nonempty'] else 1):
        raise RuntimeError('False-alarm Dice convention differs')


def normalized(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:normalized(v) for k,v in value.items()}
    if isinstance(value,list):return [normalized(v) for v in value]
    return value


def audit(reference,output,partial=False):
    output=validate_output(output);output.mkdir(parents=True,exist_ok=False)
    identity_path=reference/'identity.json';identity=json.loads(identity_path.read_text());ih=digest(identity_path)
    _guard_sources(identity['sources'])
    plan=json.loads((reference/'case_plan_frozen.json').read_text())
    if digest(reference/'case_plan_frozen.json')!=identity['case_plan_sha256']:
        raise RuntimeError('Frozen plan hash differs')
    protocol=json.loads((reference/'protocol_frozen.json').read_text())
    if protocol!=json.loads((ROOT/'protocols/mask_memory_value_headroom_v1.json').read_text()):
        raise RuntimeError('Frozen protocol differs from the source-bound protocol')
    expected={c['id']:c for c in plan['cases']}
    if len(expected)!=28 or len(plan['sequences'])!=10:raise RuntimeError('Fixed28-case/10-sequence scope differs')
    paths=list((reference/'cases').glob('*.done.json'));rows=[]
    for receipt_path in paths:
        receipt=json.loads(receipt_path.read_text());path=receipt_path.with_name(receipt_path.name.replace('.done.json','.json'))
        if receipt['identity_sha256']!=ih or receipt['result_sha256']!=digest(path):raise RuntimeError('Case receipt hash differs')
        c=json.loads(path.read_text())
        if c['id'] not in expected or any(c[k]!=v for k,v in expected[c['id']].items()):raise RuntimeError('Case selection/target identity differs')
        if c['identity_sha256']!=ih or set(c['arms'])!=set(ARMS):raise RuntimeError('Case source/arm identity differs')
        if not c['native_bank_unchanged'] or not c['native_after_oracle_exact']:raise RuntimeError('Bank/replay invariant failed')
        spans=c['spans'];target=c['target_local_frame']
        if (len(spans)!=7 or spans[0]['frame']!=0 or [s['start'] for s in spans]!=list(range(0,28672,4096))
            or [s['stop'] for s in spans]!=list(range(4096,28673,4096))
            or len({s['frame'] for s in spans})!=7 or any(s['frame']>=target for s in spans)):
            raise RuntimeError('Native full-capacity causal span coverage differs')
        if [s['local_source_frame'] for s in c['sources']]!=[s['frame'] for s in spans[1:]]:
            raise RuntimeError('GT source teacher order differs from native spatial blocks')
        if any(not s['native_reencoding_exact'] for s in c['sources']):raise RuntimeError('Source re-encoding parity failed')
        baseline=c['arms']['native'];identical=c['arms']['identity_values']
        if baseline['output_sha256']!=identical['output_sha256'] or baseline['output_sha256']!=c['native_propagation_sha256']:
            raise RuntimeError('Native/identity/propgation output parity differs')
        if any(baseline[k]!=identical[k] for k in ('scores','prediction_nonempty','object_score')):
            raise RuntimeError('Identity metric/occupancy/presence parity differs')
        for arm,q in c['arms'].items():
            check_scores(q,c['present'])
            if [k['layer'] for k in q['keys']]!=[0,1,2,3] or q['keys']!=baseline['keys'] or q['calls']!=baseline['calls']:
                raise RuntimeError('Native keys/positions/pointers or four-layer coverage differs')
            if len(q['calls'])!=1 or q['calls'][0]['spatial_keys']!=28672 or q['calls'][0]['memory_frames']!=spans:
                raise RuntimeError('Read spans differ from case teachers')
            if arm!='native':
                if [v['layer'] for v in q['values']]!=[0,1,2,3]:raise RuntimeError('Value layer coverage differs')
                if any(v['spatial_tokens']!=28672 or not v['pointer_tail_bit_identical'] or
                    v['pointer_input_sha256']!=v['pointer_output_sha256'] for v in q['values']):
                    raise RuntimeError('Spatial/pointer intervention invariant failed')
                if [(s['layer'],s['frame_block']) for s in q['value_changes']]!=[(l,b) for l in range(4) for b in range(7)]:
                    raise RuntimeError('Value statistics omit or reorder frame/layer blocks')
                if any(s['delta_rms']!=0 for s in q['value_changes'] if s['frame_block']==0 or arm=='identity_values'):
                    raise RuntimeError('Oracle changed prompt or identity values')
        rows.append(c)
    if len({c['id'] for c in rows})!=len(rows):raise RuntimeError('Duplicate case receipts')
    present_ids={c['id'] for c in rows}
    report={'scope':'Independent saved-receipt coverage/invariant/aggregation audit',
        'auditor_sha256':digest(Path(__file__)),'reference_identity_sha256':ih,
        'verified_cases':len(rows),'planned_cases':28,'missing_cases':sorted(set(expected)-present_ids),
        'four_layer_key_reads_verified':len(rows)*16,'pointer_value_reads_verified':len(rows)*12,
        'native_source_reencodings_verified':len(rows)*6,'partial':partial,
        'mask_metrics_independently_rescored':False,'efficacy_authorized':False}
    if not partial:
        if {p.stem for p in (reference/'cases').glob('*.json') if not p.name.endswith('.done.json')}!=present_ids:
            raise RuntimeError('Unreceipted or unexpected case result exists')
        completion=json.loads((reference/'completion.json').read_text());result=json.loads((reference/'results.json').read_text())
        if present_ids!=set(expected) or not completion['passed'] or completion['results_sha256']!=digest(reference/'results.json'):
            raise RuntimeError('Study is incomplete or final result hash differs')
        if set(result['case_ids'])!=set(expected) or set(result['case_receipts'])!=set(expected):raise RuntimeError('Final case coverage differs')
        for case_id,h in result['case_receipts'].items():
            if digest(reference/'cases'/f'{case_id}.json')!=h:raise RuntimeError('Final case checksum differs')
        ordered=[next(c for c in rows if c['id']==case_id) for case_id in result['case_ids']]
        recomputed=summarize(ordered,protocol['seed'])
        if normalized(recomputed)!=normalized(result['summary']):raise RuntimeError('Final summary differs from case receipts')
        report['final_results_sha256']=digest(reference/'results.json');report['complete_coverage_and_summary_match']=True
        report['primary_headroom_screen_passed']=recomputed['primary_headroom_screen_passed']
        report['decision_conditions']=recomputed['primary_decision_conditions']
    _guard_sources(identity['sources'])
    report['passed']=True
    atomic_json(output/'audit_results.json',report)
    atomic_json(output/'completion.json',{'passed':True,'audit_results_sha256':digest(output/'audit_results.json'),
        'scope':'partial audit' if partial else 'complete audit','efficacy_authorized':False})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    existed=a.output_dir.exists()
    try:print(json.dumps(audit(a.reference,a.output_dir,a.partial),indent=2))
    except Exception as error:
        if not existed and a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise
