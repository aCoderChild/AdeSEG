#!/usr/bin/env python3
"""Render independently audited Lean stored-token observations."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 data=json.loads((a.audit/'results.json').read_text())
 if not data['passed']:raise ValueError('Audit did not pass')
 lines=['# Lean stored spatial-memory observations','',
  'Exact indexed Lean replays on four selected sequences. The sequence-23 replay reproduces the archived pre-crop reference, so it includes the inset and is used only to inspect that historical run; future candidate comparisons must use the no-inset preprocessing on both arms. Counts are sampled before each non-initial memory read and count represented feature tokens, including pending dense tensors. The insurance-bank count is separate. Object pointers, position tensors, cached masks, anchors, other state and temporary tensors are outside these feature counts. Representations can alias; these are not unique allocations, bytes or a bound on total RAM.','',
  '| Sequence | Reads | Spatial feature tokens range | Insurance feature tokens range | Maximum spatial + insurance | Reads where physical exceeds author logical | Stored pointer elements range |',
  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for name,row in data['sequences'].items():
  b=row['phases']['before'];x=b['physical_range'];z=b['insurance_range'];ptr=b['pointer_elements_range']
  lines.append(f"| {name} | {b['reads']} | {x[0]}–{x[1]} | {z[0]}–{z[1]} | {b['max_represented_plus_insurance']} | {b['physical_exceeds_author_logical']} / {b['reads']} | {ptr[0]}–{ptr[1]} |")
 lines+=['','## Offline budget fit on the observed resident entries','',
  'These calculations apply the standalone whole-entry recency policy to each existing pre-read snapshot, protecting the full conditioning entry and reserving the current insurance entries. They do not simulate a new trajectory, retained history, eviction, timing of writes or a new model. An early failure means the protected prompt, bank and newest entry cannot fit.','',
  '| Sequence | Budget | Read snapshots checked | All existing history fits | Protected entries cannot fit | Sum of entries excluded across snapshots |', '| --- | ---: | ---: | ---: | ---: | ---: |']
 for name,row in data['sequences'].items():
  for budget,x in row['phases']['before']['offline_resident_snapshot_what_if'].items():
   lines.append(f"| {name} | {budget} | {x['reads_checked']} | {x['all_resident_history_fits']} | {x['protected_allocation_does_not_fit']} | {x['summed_resident_entries_excluded']} |")
 lines+=['','The author currently prunes some older visual entries after history fusion, so a selector cannot extend the history from these snapshots alone. The fixed frame-slot read path, fusion, insurance and pointer behavior all require explicit integration and validation. No accuracy improvement or deployed token cap is established here. Two independent seq3 gates reproduced the reference and one another, and the audit verified all saved candidate and reference masks, selections and indexed-position receipts. The token policy and observer tests passed.']
 a.output.write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
