#!/usr/bin/env python3
"""Compare saved per-read spatial and pointer key counts in two runs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
def read(p):return json.loads(p.read_text())
def load(path):
 obs=read(path/'token_window_observation.json')
 if not obs['passed']:raise ValueError('Observation invalid')
 if (path/'records.json').exists():
  rec=read(path/'records.json');expected=rec['positions_sha256']
  if digest(path/'records.json')!=obs['records_sha256']:raise ValueError('Records changed')
 else:
  rec=read(path/'completion.json');expected=rec['positions_sha256']
  if digest(path/'frames.json')!=obs['records_sha256']:raise ValueError('Frames changed')
 if digest(path/'positions.json')!=expected:raise ValueError('Position trace changed')
 return obs,read(path/'positions.json')
def main():
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--require-equal',action='store_true');a=p.parse_args()
 x,ar=load(a.first);y,br=load(a.second)
 if len(ar)!=len(br) or x['sequence']!=y['sequence']:raise ValueError('Trace scope differs')
 rows=[{'read':i,'first_spatial':a['spatial_keys'],'second_spatial':b['spatial_keys'],'first_pointer':a['pointer_keys'],'second_pointer':b['pointer_keys']} for i,(a,b) in enumerate(zip(ar,br))]
 spatial=sum(r['first_spatial']==r['second_spatial'] for r in rows);pointers=sum(r['first_pointer']==r['second_pointer'] for r in rows)
 if a.require_equal and (spatial!=len(rows) or pointers!=len(rows)):raise ValueError('Attention key counts are not matched')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'passed':True,'sequence':x['sequence'],'reads':len(rows),'matching_spatial_counts':spatial,'matching_pointer_counts':pointers,
         'first_observation_sha256':digest(a.first/'token_window_observation.json'),'second_observation_sha256':digest(a.second/'token_window_observation.json'),'source_sha256':digest(Path(__file__)),'rows':rows}
 (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
