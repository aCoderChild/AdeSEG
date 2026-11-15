#!/usr/bin/env python3
"""Bind corrected V3 metric audit to an unchanged V2 batch and failed V2 log."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import DATA,EXPERIMENT_ROOT
from scripts.score_additional_author_comparators_v2 import validate_frozen
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def require(value,message):
    if not value:raise ValueError(message)


def read_bound(path,sha):
    path=Path(path);require(digest(path)==sha,'Artifact changed before read')
    import hashlib
    raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==sha and digest(path)==sha,'Artifact changed during read')
    return json.loads(raw)


def reconcile(batch,audit_path):
    batch=Path(batch).resolve(strict=True);audit_path=Path(audit_path).resolve(strict=True)
    require(batch.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)) and audit_path.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)),
            'Inputs outside experiment folder')
    paths={'manifest':batch/'run_manifest.json','run_map':batch/'run_map.json','phase_log':batch/'phases.jsonl',
           'verification':batch/'verification/verification.json','scores':batch/'scores/results.json',
           'failed_log':batch/'audit.log','v2_auditor':ROOT/'scripts/audit_additional_author_comparators_v2.py',
           'v3_auditor':ROOT/'scripts/audit_additional_author_comparators_v3.py','corrected_audit':audit_path}
    sha={key:digest(path) for key,path in paths.items()}
    failed=paths['failed_log'].read_text()
    require(digest(paths['failed_log'])==sha['failed_log'] and "metadata[seq]" in failed and
            'TypeError: list indices must be integers or slices, not str' in failed,
            'Expected V2 metadata-index failure is not preserved')
    corrected=read_bound(audit_path,sha['corrected_audit']);verification=read_bound(paths['verification'],sha['verification'])
    score=read_bound(paths['scores'],sha['scores']);manifest=read_bound(paths['manifest'],sha['manifest'])
    phase_rows=[json.loads(s) for s in paths['phase_log'].read_text().splitlines()]
    require(digest(paths['phase_log'])==sha['phase_log'] and phase_rows and phase_rows[-1]['phase']=='score' and
            all(x.get('passed') is True for x in phase_rows) and not any(x.get('phase')=='audit' for x in phase_rows),
            'Original V2 phase log is not the preserved scored-but-unaudited state')
    phase_artifacts={row['phase']:row.get('artifact_sha256') for row in phase_rows}
    require(phase_artifacts.get('verify')==sha['verification'] and phase_artifacts.get('score')==sha['scores'],
            'Original V2 phase-frozen verification or score differs')
    require(corrected.get('passed') is True and corrected.get('error_count')==0 and not corrected.get('errors') and
            corrected.get('auditor_sha256')==sha['v3_auditor'] and corrected.get('run_map_sha256')==sha['run_map'] and
            corrected.get('verification_sha256')==sha['verification'] and corrected.get('score_sha256')==sha['scores'] and
            corrected.get('prelabel_guard',{}).get('open_attempts')==0 and
            score.get('passed') is True and score.get('verification_sha256')==sha['verification'] and
            score.get('run_map_sha256')==sha['run_map'] and verification.get('passed') is True and
            verification.get('label_guard',{}).get('open_attempts')==0 and
            manifest.get('sources',{}).get('auditor')==sha['v2_auditor'],
            'Corrected audit, original score or original source binding differs')
    expected_labels=sum(len(verification['inputs'][s]['images']) for s in verification['selection'])
    expected_predictions=sum(len(verification['inputs'][s]['images'])*(3+len(verification['references'][s]))
                             for s in verification['selection'])
    require(corrected.get('label_files_checked')==expected_labels and corrected.get('prediction_files_checked')==expected_predictions,
            'Corrected audit frame/prediction coverage differs')
    current=validate_frozen(paths['run_map'],allow_completed_score=True)
    for key in ('selection','run_map_sha256','launch_manifest_sha256','receipts','references','inputs'):
        require(current[key]==verification[key],'Original frozen predictions changed')
    require(all(digest(path)==sha[key] for key,path in paths.items()),'Bound source or evidence changed during reconciliation')
    return {'passed':True,'scope':'V2 auditor failed on metadata-list indexing; separate corrected V3 audit rescored the same frozen outputs.',
            'batch':str(batch),'batch_run_manifest_sha256':sha['manifest'],'batch_run_map_sha256':sha['run_map'],
            'batch_phase_log_sha256':sha['phase_log'],'batch_verification_sha256':sha['verification'],
            'batch_scores_sha256':sha['scores'],'failed_audit_log_sha256':sha['failed_log'],
            'failed_audit_source_sha256':sha['v2_auditor'],'corrected_audit_path':str(audit_path),
            'corrected_audit_sha256':sha['corrected_audit'],'corrected_auditor_sha256':sha['v3_auditor'],
            'reconciler_sha256':digest(Path(__file__)),'expected_labels':expected_labels,'expected_predictions':expected_predictions}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('batch','corrected-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    require(out.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)) and not out.exists() and out.suffix=='.json',
            'Output must be a new JSON file under Drive experiments')
    guard=install_mask_open_guard(DATA);result=reconcile(a.batch,a.corrected_audit)
    require(guard['open_attempts']==0,'Annotation opened during reconciliation');result['label_guard']=dict(guard)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'output':str(out)}))


if __name__=='__main__':main()
