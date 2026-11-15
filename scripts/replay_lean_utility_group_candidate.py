#!/usr/bin/env python3
"""Run matched group replays only after the independent engineering gate."""
import argparse,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
p=argparse.ArgumentParser(add_help=False);p.add_argument('--deletion-verification',type=Path,required=True)
a,rest=p.parse_known_args()
if '--output' not in rest or '--capture' not in rest:raise ValueError('Capture and output required')
out=Path(rest[rest.index('--output')+1]);capture=Path(rest[rest.index('--capture')+1]);v=json.loads((a.deletion_verification/'verification.json').read_text())
if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_utility_group_deletion.py') or not v['eligible_reads']:raise ValueError('Deletion engineering verification required')
first=a.deletion_verification.parent/'gate_attempt01/completion.json';second=a.deletion_verification.parent/'gate_attempt02/completion.json'
if digest(first)!=v['first_sha256'] or digest(second)!=v['second_sha256']:raise ValueError('Verified gate changed')
g=json.loads(first.read_text())
if g['protocol_sha256']!=digest(ROOT/'protocols/lean_utility_group_deletion_v1.json') or g['runner_sha256']!=digest(ROOT/'scripts/replay_lean_utility_group_deletion.py') or g['partition_sha256']!=digest(ROOT/'modeling/lean_utility_group_deletion.py'):raise ValueError('Deletion design or implementation changed')
if json.loads((capture/'utility_capture_receipt.json').read_text())['sequence'] not in ('seq2','seq4','seq9','seq10'):raise ValueError('Candidate outside approved split')
old=sys.argv
try:
 sys.argv=[str(ROOT/'scripts/replay_lean_utility_group_deletion.py'),*rest];runpy.run_path(sys.argv[0],run_name='__main__')
finally:sys.argv=old
(out/'candidate_receipt.json').write_text(json.dumps({'passed':True,'completion_sha256':digest(out/'completion.json'),'gate_verification_sha256':digest(a.deletion_verification/'verification.json'),'runner_sha256':digest(Path(__file__))},indent=2)+'\n')
