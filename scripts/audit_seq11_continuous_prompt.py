#!/usr/bin/env python3
"""Independently verify dense prompts and rescore saved seq11 outputs."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.continuous_mask_prompt import mask_and_score_prompts
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('pilot','gate-a','gate-b','capture','reference','prior-audit','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();data=read(a.pilot/'predictions.json');prior=read(a.prior_audit/'results.json');ref=read(a.reference/'records.json');done=read(a.reference/'completion.json');cap=read(a.capture/'capture.json');ident=data['identity']
 if not done['passed'] or digest(a.reference/'records.json')!=done['records_sha256'] or ident['reference_sha256']!=done['records_sha256'] or ident['capture_sha256']!=digest(a.capture/'capture.json') or ident['source_sha256']!=digest(ROOT/'scripts/run_seq11_continuous_prompt.py') or ident['protocol_sha256']!=digest(ROOT/'protocols/seq11_continuous_mask_prompt_v1.json') or ident['prompt_sha256']!=digest(ROOT/'modeling/continuous_mask_prompt.py') or prior['audit_sha256']!=digest(ROOT/'scripts/audit_seq11_temporal_refinement.py') or not read(a.pilot/'parameter_integrity_after.json')['unchanged']:
  raise ValueError('Pilot source invalid')
 gates=[]
 for path in (a.gate_a,a.gate_b):
  g=read(path/'gate.json')
  if g['identity']!=ident or not read(path/'parameter_integrity_after.json')['unchanged']:raise ValueError('Gate identity invalid')
  for item in g['records']:
   for arm,h in item['masks'].items():
    if digest(path/f"{item['target']:06d}_{arm}.png")!=h:raise ValueError('Gate mask changed')
   if next(x for x in data['records'] if x['target']==item['target'])!=item:raise ValueError('Pilot differs from gate')
  gates.append(g)
 if gates[0]!=gates[1]:raise ValueError('Gates differ')
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq11'];priorrows={x['target']:x for x in prior['rows']};rows=[]
 for item in data['records']:
  src,i=item['source'],item['target'];earlier=priorrows[i];source=next(x for x in cap['outputs'] if x['source']==src and x['target']==i);maskpath=a.reference/f'{i:06d}.png';scorepath=a.capture/f'{src:06d}_{i:06d}.npz';frame=frames[i]
  if digest(maskpath)!=ref['mask_sha256'][i] or digest(maskpath)!=item['input_mask_sha256'] or digest(scorepath)!=source['map_sha256'] or digest(scorepath)!=item['map_sha256'] or digest(frame.image_path)!=item['image_sha256'] or digest(frame.mask_path)!=earlier['label_sha256']:raise ValueError('Pilot input changed')
  mask=np.asarray(Image.open(maskpath))>0;seed=int.from_bytes(hashlib.sha256(f'20261114|seq11|{src}|{i}'.encode()).digest()[:8],'little')
  with np.load(scorepath,allow_pickle=False) as z:
   fg=mask_and_score_prompts(mask,z['foreground'],seed);co=mask_and_score_prompts(mask,z['contrast'],seed);rgb=mask_and_score_prompts(mask,z['rgb_contrast'],seed)
  prompts={'mask_only':co['mask_only'],'foreground':fg['score'],'contrast':co['score'],'rgb_contrast':rgb['score'],'shuffled_contrast':co['shuffled']}
  if any(hashlib.sha256(x.tobytes()).hexdigest()!=item['prompt_sha256'][arm] for arm,x in prompts.items()):raise ValueError('Dense prompt does not reproduce')
  truth=load_label_mask(frame.mask_path,[1])==1;present=bool(truth.any());metrics={'lean_reference':earlier['metrics']['lean_reference'],'box_only':earlier['metrics']['box_only'],'point_contrast':earlier['metrics']['contrast_points']}
  for arm,h in item['masks'].items():
   path=a.pilot/f'{i:06d}_{arm}.png'
   if digest(path)!=h:raise ValueError('Prediction changed')
   pred=np.asarray(Image.open(path))>0
   if pred.shape!=truth.shape:raise ValueError('Geometry differs')
   den=int(pred.sum()+truth.sum());metrics[arm]={'dice':2*int((pred&truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())}
  if present!=earlier['present']:raise ValueError('Label differs from prior audit')
  rows.append({'source':src,'target':i,'present':present,'metrics':metrics})
 if len(rows)!=len(priorrows):raise ValueError('Pilot coverage differs')
 summary={}
 for arm in rows[0]['metrics']:
  pos=[x['metrics'][arm]['dice'] for x in rows if x['present']];neg=[x['metrics'][arm]['false_alarm'] for x in rows if not x['present']]
  summary[arm]={'mean_present_dice':float(np.mean(pos)),'present_count':len(pos),'empty_false_alarms':sum(neg),'empty_count':len(neg)}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Seven selected seq11 targets; fixed synthetic binary mask prior plus continuous temporal scores. No propagation change.',
         'summary':summary,'rows':rows,'pilot_sha256':digest(a.pilot/'predictions.json'),'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Sequence 11: continuous temporal mask-prompt pilot','',result['scope'],'','| Arm | Mean present Dice | Empty false alarms |','| --- | ---: | ---: |']
 for name,x in summary.items():lines.append(f"| {name} | {x['mean_present_dice']:.6f} | {x['empty_false_alarms']} / {x['empty_count']} |")
 lines+=['','Two label-free gates reproduced all prompt and mask hashes, and the audit reconstructed each dense prompt independently before rescoring. The mask prior is a fixed linear transformation of a saved binary prediction, not the original SAM logits. Scores are added continuously at a fixed scale, without a new threshold or confidence gate.','', 'These frames and this method were chosen after prior inspection. This is a fixed-frame SAM2 image-decoder diagnostic, not a full-video modification, an independent evaluation, or proof of generalization.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
