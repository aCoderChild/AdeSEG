#!/usr/bin/env python3
"""Rescore saved same-read fusion predictions after all inference completes."""
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
    for key in ('capture','first','second','reference','merged','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();c=read(a.capture/'fusion_capture_receipt.json');one=read(a.first/'completion.json');two=read(a.second/'completion.json')
    protocol=read(ROOT/'protocols/lean_same_read_fusion_priority_v1.json')
    if c.get('targets')!=protocol['frames'].get(c.get('sequence')) or c.get('protocol_sha256')!=digest(ROOT/'protocols/lean_same_read_fusion_priority_v1.json'):
        raise ValueError('Predeclared sequence or frames changed')
    if not c['passed'] or c['entry']!='token' or c['viewport'] or c['sequence'] not in ('seq8','seq11','seq20') or not one['passed'] or not two['passed'] or one!=two:
        raise ValueError('Capture or independent replays invalid')
    if one['capture_sha256']!=digest(a.capture/'fusion_capture_receipt.json') or one['runner_sha256']!=digest(ROOT/'scripts/replay_lean_same_read_fusion_priority.py') or one['model_capture_sha256']!=digest(ROOT/'modeling/lean_same_read_fusion.py'):
        raise ValueError('Replay code or capture changed')
    if c['capture_sha256']!=digest(ROOT/'modeling/lean_same_read_fusion.py') or c['runner_sha256']!=digest(ROOT/'scripts/capture_lean_same_read_fusion_priority.py') or c['output_receipt_sha256']!=digest(a.capture/'fused_control_receipt.json'):
        raise ValueError('Capture provenance changed')
    for folder in (a.capture,a.first,a.second):
        integrity=read(folder/'parameter_integrity_after.json')
        if not integrity['unchanged'] or integrity['before']!=integrity['after'] or integrity['changed_names']:
            raise ValueError('Model parameters changed')
    seq=c['sequence'];original=read(a.reference/'completion.json');reference_records=read(a.reference/'records.json')
    location=Path(read(a.merged/'locations.json')[seq]);merged_done=read(location/'completion.json');merged_record=read(location/'records.json')[0];rows=read(location/seq/'frames.json')
    if not original['passed'] or original['records_sha256']!=digest(a.reference/'records.json') or reference_records['sequence']!=seq or not merged_done['passed'] or digest(location/'records.json')!=merged_done['records_sha256'] or digest(location/seq/'frames.json')!=merged_record['frames_sha256'] or len(rows)!=reference_records['frames']:raise ValueError('Reference audit input changed')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq]
    events=[]
    for event,snapshot in zip(one['events'],c['snapshots']):
        i=event['frame']
        if i!=snapshot['frame'] or digest(a.capture/'fusion_sources'/f'{i:06d}.pt')!=snapshot['source_sha256'] or digest(a.capture/'snapshots'/f'{i:06d}.pt')!=snapshot['read_sha256']:
            raise ValueError('Capture tensors changed')
        if digest(frames[i].mask_path)!=rows[i]['label_sha256']:raise ValueError('Reference label changed')
        truth=load_label_mask(frames[i].mask_path,[1])==1;present=bool(truth.any());masks={};scores={}
        for method,sha in event['masks'].items():
            p1=a.first/f'{i:06d}_{method}.png';p2=a.second/f'{i:06d}_{method}.png'
            if digest(p1)!=sha or digest(p2)!=sha:raise ValueError('Saved repeat mask changed')
            pred=np.asarray(Image.open(p1))>0
            if pred.shape!=truth.shape:raise ValueError('Prediction geometry differs')
            den=int(pred.sum()+truth.sum());scores[method]={'dice':2*int((pred&truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())};masks[method]=pred
        if set(masks)!={'noop','fused_reconstructed','raw_skeleton'} or not np.array_equal(masks['noop'],masks['fused_reconstructed']):raise ValueError('Fused control changed')
        if not np.array_equal(masks['noop'],np.asarray(Image.open(a.reference/f'{i:06d}.png'))>0):raise ValueError('No-op does not match fused reference')
        events.append({'frame':i,'present':present,'label_sha256':rows[i]['label_sha256'],'scores':scores,'raw_changed_pixels':int(np.count_nonzero(masks['noop']^masks['raw_skeleton'])),
          'skeleton_frame':event['source_frame'],'alpha':event['alpha'],'raw_fused_mean_abs':event['raw_fused_mean_abs'],'spatial_keys':event['spatial_keys'],'pointer_keys':event['pointer_keys']})
    if {e['frame'] for e in events}!=set(c['targets']):raise ValueError('Frame coverage differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'sequence':seq,'scope':'Retrospective fixed-read raw skeleton versus author-fused skeleton; no changed video trajectory.',
      'frames':events,'capture_sha256':digest(a.capture/'fusion_capture_receipt.json'),'replay_sha256':digest(a.first/'completion.json'),
      'repeat_sha256':digest(a.second/'completion.json'),'reference_frames_sha256':digest(location/seq/'frames.json'),'audit_sha256':digest(Path(__file__))}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=[f'# {seq}: same-read raw versus fused memory','','The author’s fused second-newest memory skeleton is replaced by its own raw skeleton from the same captured state. Query, other memory, positional embeddings, pointers, and key count are unchanged. No inference result is fed into later frames.','',
      '| Frame | Label | Fused Dice | Raw-skeleton Dice | Raw − fused | Changed pixels | Empty false alarm fused / raw |','| ---: | :--- | ---: | ---: | ---: | ---: | :--- |']
    for e in events:
        f=e['scores']['noop'];r=e['scores']['raw_skeleton'];empty='—' if e['present'] else f"{f['false_alarm']} / {r['false_alarm']}"
        lines.append(f"| {e['frame']} | {'present' if e['present'] else 'empty'} | {f['dice']:.6f} | {r['dice']:.6f} | {r['dice']-f['dice']:+.6f} | {e['raw_changed_pixels']} | {empty} |")
    lines+=['',"Retrospectively selected frames; this estimates only a local conditional change. The author's pre-existing reliability weighting remains part of the fused value; no new inference gate was introduced. These observations cannot establish how a raw-history model would propagate or generalize.",'']
    (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print(str(out/'report.md'))
if __name__=='__main__':main()
