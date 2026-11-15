#!/usr/bin/env python3
"""Check admitted token-window frame IDs against an observed six-frame window."""
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('root','audit','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();d=read(a.audit/'completion.json');data=read(a.audit/'results.json')
 if not d['passed'] or digest(a.audit/'results.json')!=d['results_sha256'] or d['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py'):raise ValueError('Independent audit invalid')
 results={}
 for name in ('seq8','seq11','seq20','seq23'):
  run=a.root/f'{name}_attempt01';obs=read(run/'token_window_observation.json');path=run/'token_window_events.json'
  if not obs['passed'] or digest(path)!=obs['events_sha256']:raise ValueError('Events changed')
  plans=[x for x in read(path) if x['stage']=='admission_plan']
  if not plans:raise ValueError('Missing admissions')
  first=plans[0]['frame'];matches=0;lengths=collections.Counter();mismatches=[]
  for plan in plans:
   ids=plan['selected_frames'];expected=list(range(max(first,plan['frame']-5),plan['frame']+1));lengths[len(ids)]+=1
   if ids==expected:matches+=1
   else:mismatches.append({'frame':plan['frame'],'actual':ids,'six_most_recent':expected})
  results[name]={'admissions':len(plans),'same_as_six_latest':matches,'mismatches':mismatches,'retained_count_distribution':dict(sorted(lengths.items())),
                 'events_sha256':obs['events_sha256']}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Compare selected frame IDs only; does not equate token contents, fusion or predictions with a six-frame control.',
         'sequences':results,'audit_results_sha256':d['results_sha256'],'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Token-window admitted-frame analysis','',result['scope'],'','| Sequence | Admissions | IDs equal the latest six available frames | Retained-count distribution |','| --- | ---: | ---: | --- |']
 for name,x in results.items():lines.append(f"| {name} | {x['admissions']} | {x['same_as_six_latest']} / {x['admissions']} | {x['retained_count_distribution']} |")
 lines+=['','For these runs, conditioning and newest full grids use 4,096 entries each, insurance capacity reserves 3,072, and retained prior grids generally hold 1,024 each. At the selected 16,384 budget, up to five such prior grids fit. The selector has no fixed frame-count limit, but the observed IDs happen to match six-frame recency. This does not establish that changing the number of frames contributed to the seq23 improvement. The original Lean baseline also uses fused history, so matching frame IDs does not make the systems equivalent.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
