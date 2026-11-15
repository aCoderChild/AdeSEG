#!/usr/bin/env python3
"""Exercise ephemeral CPU scalar to MPS transfer with and without blocking."""
import argparse,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
 if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise ValueError('Explicitly disable MPS fallback')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 rows=[]
 for nonblocking in (True,False):
  bad=0;count=0
  for _ in range(20):
   copied=[torch.tensor([j]).to('mps',non_blocking=nonblocking) for j in range(2000)]
   actual=torch.cat(copied).cpu();expected=torch.arange(2000)
   bad+=int((actual!=expected).sum());count+=len(expected)
  rows.append({'nonblocking':nonblocking,'errors':bad,'total':count})
 result={'scope':'ephemeral integer CPU tensor lifetime stress, not a segmentation test',
  'torch_version':torch.__version__,'fallback':os.environ['PYTORCH_ENABLE_MPS_FALLBACK'],
  'runner_sha256':digest(Path(__file__)),'rows':rows}
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
