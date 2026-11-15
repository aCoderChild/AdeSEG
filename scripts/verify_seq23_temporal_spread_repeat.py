#!/usr/bin/env python3
"""Verify independent full cropped-seq23 token-window replay artifacts."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('first','repeat','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();files=('frames.json','prompt.json','positions.json','selections.json','events.json','token_window_events.json');observations=[]
 for run in (a.first,a.repeat):
  o=read(run/'token_window_observation.json');done=read(run/'completion.json')
  if not o['passed'] or o['mode']!='candidate' or o['sequence']!='seq23' or o['source_sha256']!=digest(ROOT/'scripts/run_seq23_temporal_spread_window.py') or not done['passed'] or digest(run/'frames.json')!=done['frames_sha256'] or digest(run/'frames.json')!=o['records_sha256'] or digest(run/'token_window_events.json')!=o['events_sha256'] or not read(run/'parameter_integrity_after.json')['unchanged']:
   raise ValueError('Run invalid')
  observations.append(o)
 if observations[0]!=observations[1]:raise ValueError('Run identity or event receipts differ')
 hashes={}
 for f in files:
  hashes[f]=digest(a.first/f)
  if digest(a.repeat/f)!=hashes[f]:raise ValueError('Trace differs: '+f)
 rows=read(a.first/'frames.json');other=read(a.repeat/'frames.json')
 if len(rows)!=len(other):raise ValueError('Frame counts differ')
 for i,(x,y) in enumerate(zip(rows,other)):
  if x['mask_sha256']!=y['mask_sha256'] or digest(a.first/f'{i:06d}.png')!=x['mask_sha256'] or digest(a.repeat/f'{i:06d}.png')!=x['mask_sha256']:
   raise ValueError('Mask differs: '+str(i))
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'passed':True,'sequence':'seq23','frames':len(rows),'all_masks_equal':True,'trace_sha256':hashes,
         'first_observation_sha256':digest(a.first/'token_window_observation.json'),'repeat_observation_sha256':digest(a.repeat/'token_window_observation.json'),'source_sha256':digest(Path(__file__))}
 (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'frames':len(rows)},indent=2))
if __name__=='__main__':main()
