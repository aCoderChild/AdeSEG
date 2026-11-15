#!/usr/bin/env python3
"""Independently audit matched-budget conditioning-memory selection."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('usage','uniform','merged','historical','indexed-root','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();v=read(a.verification);proto=read(ROOT/'protocols/lean_condition_memory_pruning_v1.json');merge=read(a.merged/'completion.json');old_audit=read(a.indexed_root/'audited_cohort_attempt01/completion.json')
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_condition_memory_pruning.py') or not merge['passed']
     or digest(a.merged/'results.json')!=merge['results_sha256'] or digest(a.historical/'comparison.json')!=merge['historical_comparison_sha256']
     or not old_audit['passed'] or digest(a.indexed_root/'audited_cohort_attempt01/results.json')!=old_audit['results_sha256']):raise ValueError('Reference verification invalid')
 historical={x['sequence']:x for x in read(a.historical/'comparison.json')['sequences']};reference=read(a.indexed_root/'audited_cohort_attempt01/results.json')['sequences'];loc=read(a.merged/'locations.json')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};parts={}
 for name in ('usage','uniform'):
  part=getattr(a,name);setup=read(part/'setup.json');done=read(part/'completion.json')
  if (not done['passed'] or done['sequences']!=setup['sequences'] or set(setup['sequences'])!=set(proto['cohort'])
      or setup['strategy']!=name or setup['verification_sha256']!=digest(a.verification) or setup['runner_sha256']!=v['identity']['runner_sha256']):raise ValueError('Partition invalid')
  parts[name]=part
 results={}
 for name in proto['cohort']:
  refs=[x/name for x in (a.indexed_root/'part_a_attempt01',a.indexed_root/'part_b_attempt01') if (x/name/'completion.json').exists()]
  if len(refs)!=1:raise ValueError('Reference missing')
  ref=refs[0];rr=read(ref/'records.json');rd=read(ref/'completion.json');rrs=read(ref/'frames.json');refpos=read(ref/'positions.json')
  if not rd['passed'] or digest(ref/'records.json')!=rd['records_sha256'] or digest(ref/'frames.json')!=rr['frames_sha256'] or digest(ref/'positions.json')!=rr['positions_sha256']:raise ValueError('Reference changed')
  original=Path(loc[name]);old=read(original/'records.json')[0];olddone=read(original/'completion.json');oldaudit=read(original/'independent_audit.json');folder=original/name
  if not olddone['passed'] or not oldaudit['passed'] or digest(original/'records.json')!=olddone['records_sha256'] or oldaudit['records_sha256']!=olddone['records_sha256'] or digest(folder/'frames.json')!=old['frames_sha256'] or digest(folder/'prompt.json')!=old['prompt_sha256']:raise ValueError('Original changed')
  oldrows=read(folder/'frames.json');first=read(folder/'prompt.json')['first_detection_index'];frames=videos[name];condition={};usages=[]
  for strategy,part in parts.items():
   run=part/name;z=read(run/'condition_intervention.json');rec=read(run/'records.json');done=read(run/'completion.json');setup=read(run/'setup.json');events=read(run/'condition_events.json')
   if (not z['passed'] or z['mode']!='study' or z['strategy']!=strategy or z['sequence']!=name or z['identity']!=v['identity'] or z['reference_records_sha256']!=rd['records_sha256']
       or z['model_records_sha256']!=digest(run/'records.json') or z['condition_events_sha256']!=digest(run/'condition_events.json')
       or not done['passed'] or done['records_sha256']!=z['model_records_sha256'] or setup['engine_variant_sha256']!=v['identity']['engine_sha256']
       or not read(run/'parameter_integrity_after.json')['unchanged']):raise ValueError('Candidate invalid')
   for key in ('positions','selections','memory_events','frames'):
    if digest(run/f'{key}.json')!=rec[key+'_sha256']:raise ValueError('Candidate trace changed')
   selected=[e for e in events if e['event']=='selection'];reads=[e for e in events if e['event']=='conditioning_read']
   if len(selected)!=1 or not reads or reads[0]['tokens']!=4096 or any(x['tokens']!=1024 for x in reads[1:]) or len(reads)!=z['condition_reads']:raise ValueError('Condition budget changed')
   sel=selected[0];usage=torch.tensor(sel['first_read_usage'],dtype=torch.float32);idx=torch.tensor(sel['selected_indices'],dtype=torch.long)
   expected=torch.topk(usage,1024).indices if strategy=='usage' else torch.arange(4096).reshape(64,64)[::2,::2].reshape(-1)
   if usage.numel()!=4096 or not torch.equal(idx,expected) or hashlib.sha256(idx.numpy().tobytes()).hexdigest()!=sel['selected_indices_sha256'] or sel['selected_indices_sha256']!=z['indices_sha256']:raise ValueError('Condition selection mismatch')
   if abs(float((usage[idx].sum()/usage.sum()).item())-sel['selected_usage_fraction_of_conditioning'])>1e-7:raise ValueError('Selected attention share differs')
   usagelist=usage.numpy();usages.append(usagelist)
   pos=read(run/'positions.json');noncond=read(run/'selections.json')
   if len(pos)!=len(refpos) or any(x['pointer_keys']!=y['pointer_keys'] for x,y in zip(pos,refpos)) or any(x['candidates']!=4096 or x['kept']!=1024 for x in noncond):raise ValueError('Pointer or noncondition budget changed')
   if rec['mask_sha256'][:first+2]!=rr['mask_sha256'][:first+2] or z['changed_masks']!=sum(x!=y for x,y in zip(rec['mask_sha256'],rr['mask_sha256'])):raise ValueError('Causal prefix or changed count differs')
   condition[strategy]={'run':run,'record':rec,'rows':read(run/'frames.json'),'selected_usage_fraction':sel['selected_usage_fraction_of_conditioning'],'condition_reads':len(reads),'changed_masks':z['changed_masks']}
  if not np.array_equal(*usages):raise ValueError('Strategies saw different first-read attention usage')
  if not(len(frames)==len(oldrows)==len(rrs)==len(rr['mask_sha256'])==len(condition['usage']['rows'])==len(condition['uniform']['rows'])):raise ValueError('Frame coverage changed')
  values=[]
  for i,(frame,oldrow,refrow) in enumerate(zip(frames,oldrows,rrs)):
   if digest(frame.image_path)!=oldrow['image_sha256'] or digest(frame.mask_path)!=oldrow['label_sha256']:raise ValueError('Input changed')
   truth=load_label_mask(frame.mask_path,[1])==1;present=bool(truth.any());scores={}
   for strategy,path,row,hashes in [('reference',ref/f'{i:06d}.png',refrow,rr['mask_sha256'])]+[(strategy,x['run']/f'{i:06d}.png',x['rows'][i],x['record']['mask_sha256']) for strategy,x in condition.items()]:
    if row['frame']!=i or digest(path)!=row['mask_sha256'] or row['mask_sha256']!=hashes[i]:raise ValueError('Mask changed')
    pred=np.asarray(Image.open(path))>0
    if pred.shape!=truth.shape:raise ValueError('Mask shape changed')
    den=int(pred.sum()+truth.sum());dice=2*int((pred&truth).sum())/den if den else 1.;alarm=bool(pred.any()) if not present else None
    if row['present']!=present or not np.isclose(row['dice'],dice,rtol=0,atol=1e-12) or row['false_alarm']!=alarm:raise ValueError('Score differs')
    scores[strategy]={'dice':dice,'false_alarm':alarm}
   values.append({'present':present,'scores':scores})
  targets=[json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()]
  if len(targets)!=historical[name]['targets'] or any(values[x['frame']]['present']!=x['present'] for x in targets):raise ValueError('Historical target changed')
  data={'frames':len(frames),'all_present':sum(x['present'] for x in values),'all_empty':sum(not x['present'] for x in values),'historical_present':sum(x['present'] for x in targets),'historical_empty':sum(not x['present'] for x in targets),
        'native_gt_dice':historical[name]['native_dice'],'lean_gt_dice':historical[name]['lean_dice']}
  for strategy in ('reference','usage','uniform'):
   present=[x['scores'][strategy]['dice'] for x in values if x['present']];empty=[x['scores'][strategy]['false_alarm'] for x in values if not x['present']]
   hp=[values[x['frame']]['scores'][strategy]['dice'] for x in targets if x['present']];he=[values[x['frame']]['scores'][strategy]['false_alarm'] for x in targets if not x['present']]
   data[strategy]={'all_dice':float(np.mean(present)),'all_false_alarms':sum(empty),'historical_dice':float(np.mean(hp)),'historical_false_alarms':sum(he)}
   if strategy!='reference':data[strategy].update({k:condition[strategy][k] for k in ('selected_usage_fraction','condition_reads','changed_masks')})
  if not np.isclose(data['reference']['historical_dice'],reference[name]['candidate_historical_dice'],rtol=0,atol=1e-12):raise ValueError('Reference audit differs')
  results[name]=data
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);rows=list(results.values());summary={'scope':'Four retrospectively inspected difficult videos, same automatic prompts. Usage and fixed spatial uniform controls both retain 1,024 of 4,096 conditioning tokens after the unchanged first read; the baseline retains all. Historical native and Lean use GT prompts.','sequences':results}
 for strategy in ('reference','usage','uniform'):
  summary[strategy]={'all_dice':float(np.mean([x[strategy]['all_dice'] for x in rows])),'all_false_alarms':sum(x[strategy]['all_false_alarms'] for x in rows),
   'historical_dice':float(np.mean([x[strategy]['historical_dice'] for x in rows])),'historical_false_alarms':sum(x[strategy]['historical_false_alarms'] for x in rows)}
 summary['all_empty']=sum(x['all_empty'] for x in rows);summary['historical_empty']=sum(x['historical_empty'] for x in rows)
 (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Fixed-budget conditioning selection: independent four-video audit','',summary['scope'],'','| Sequence | Native GT | Lean GT | Full conditioning | Usage-ranked | Uniform grid | Empty false alarms full / usage / uniform |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for name,x in results.items():
  lines.append(f"| {name} | {x['native_gt_dice']:.3f} | {x['lean_gt_dice']:.3f} | {x['reference']['historical_dice']:.3f} | {x['usage']['historical_dice']:.3f} | {x['uniform']['historical_dice']:.3f} | {x['reference']['all_false_alarms']} / {x['usage']['all_false_alarms']} / {x['uniform']['all_false_alarms']} ({x['all_empty']}) |")
 lines+=['',f"Equal-sequence matched Dice: {summary['reference']['historical_dice']:.4f} full, {summary['usage']['historical_dice']:.4f} usage, {summary['uniform']['historical_dice']:.4f} uniform. Matched labeled-empty false alarms: {summary['reference']['historical_false_alarms']} / {summary['usage']['historical_false_alarms']} / {summary['uniform']['historical_false_alarms']} of {summary['historical_empty']}.",
  'Four selected, previously inspected sequences are exploratory; no generalization claim follows. Usage and uniform are matched in conditioning-token budget, while full conditioning uses more tokens. Attention normalization and downstream memory change when the conditioning grid is pruned. No inference-time labels, new re-detection, threshold or confidence gate were used. Runtime was not measured as a controlled benchmark.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
