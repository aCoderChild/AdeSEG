#!/usr/bin/env python3
"""Rescore completed fixed-read predictions; labels are opened only here."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for name in ('capture','replay','merged','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();capture=read(a.capture/'priority_capture_receipt.json');replay=read(a.replay/'completion.json')
    if not capture['passed'] or not replay['passed'] or replay['sequence']!=capture['sequence'] or replay['capture_sha256']!=digest(a.capture/'priority_capture_receipt.json') or replay['runner_sha256']!=digest(ROOT/'scripts/replay_lean_region_priority_intervention.py') or replay['rgb_module_sha256']!=digest(ROOT/'modeling/lean_rgb_contrast_control.py'):
        raise ValueError('Prediction receipts invalid')
    seq=capture['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq]
    if seq=='seq23': rows=read(Path(capture['reference'])/'frames.json')
    else:
        source=Path(read(a.merged/'locations.json')[seq]);done=read(source/'completion.json');record=read(source/'records.json')[0]
        if digest(source/'records.json')!=done['records_sha256'] or digest(source/seq/'frames.json')!=record['frames_sha256']:raise ValueError('Original source metadata changed')
        rows=read(source/seq/'frames.json')
    for index,sha in replay['rgb_source_image_sha256'].items():
        i=int(index)
        if sha!=rows[i]['image_sha256'] or digest(frames[i].image_path)!=sha:raise ValueError('RGB source image changed')
    prompt_path=a.capture/'prompt.png' if seq=='seq23' else source/seq/'prompt_refined.png'
    if digest(prompt_path)!=replay['rgb_prompt_sha256']:raise ValueError('RGB prompt changed')
    results=[]
    for event in replay['events']:
        i=event['frame'];frame=frames[i];row=rows[i]
        if digest(frame.mask_path)!=row['label_sha256']:raise ValueError('Reference label changed')
        truth=load_label_mask(frame.mask_path,[1])==1;present=bool(truth.any());scores={}
        for method,expected in event['masks'].items():
            path=a.replay/f'{i:06d}_{method}.png'
            if digest(path)!=expected:raise ValueError('Predicted mask changed')
            pred=np.asarray(Image.open(path))>0
            if pred.shape!=truth.shape:raise ValueError('Prediction geometry differs')
            den=int(pred.sum()+truth.sum());scores[method]={'dice':2*int((pred&truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())}
        results.append({'frame':i,'present':present,'label_sha256':row['label_sha256'],'scores':scores})
    if {r['frame'] for r in results}!=set(capture['targets']):raise ValueError('Target coverage differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'sequence':seq,'scope':'Retrospective fixed-read 1024-token deletion; not a full-video or independent generalization estimate',
      'capture_receipt_sha256':digest(a.capture/'priority_capture_receipt.json'),'replay_completion_sha256':digest(a.replay/'completion.json'),
      'source_sha256':digest(Path(__file__)),'frames':results}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json')},indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':seq,'frames':len(results)}))
if __name__=='__main__':main()
