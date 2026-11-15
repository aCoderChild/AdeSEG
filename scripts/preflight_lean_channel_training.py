#!/usr/bin/env python3
"""Freeze sequence-isolated, label-free feature-capture selections for a pilot."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

PROTOCOL=ROOT/'protocols/lean_channel_attention_training_pilot_v1.json'

def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--merged',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    proto=read(PROTOCOL);groups=proto['groups'];names=[x for group in groups.values() for x in group]
    if len(names)!=len(set(names)) or set(groups['protected_priority'])!={'seq8','seq11','seq20','seq23'}:raise ValueError('Split contains overlap')
    done=read(a.merged/'completion.json');locations=read(a.merged/'locations.json')
    if not done['passed'] or digest(a.merged/'results.json')!=done['results_sha256']:raise ValueError('Merged reference changed')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    report={};hash_sets={}
    for seq in names:
        source=Path(locations[seq]);source_done=read(source/'completion.json');record=read(source/'records.json')[0];folder=source/seq
        if not source_done['passed'] or digest(source/'records.json')!=source_done['records_sha256'] or digest(folder/'frames.json')!=record['frames_sha256'] or digest(folder/'prompt.json')!=record['prompt_sha256']:
            raise ValueError('Archived sequence metadata changed: '+seq)
        rows=read(folder/'frames.json');prompt=read(folder/'prompt.json');first=prompt['first_detection_index']
        if first is None or len(rows)!=len(videos[seq]) or digest(folder/'prompt_refined.png')!=prompt['refined_mask_sha256']:raise ValueError('Prompt or video unavailable: '+seq)
        indices=sorted({first+1+round((len(rows)-1-(first+1))*i/4) for i in range(5)})
        if len(indices)!=5 or indices[0]<=first or indices[-1]>=len(rows):raise ValueError('Insufficient capture coverage: '+seq)
        for i in indices:
            if digest(videos[seq][i].image_path)!=rows[i]['image_sha256']:raise ValueError('Selected source image changed: '+seq)
        hash_sets[seq]={r['image_sha256'] for r in rows}
        report[seq]={'source':str(source),'first_prompt':first,'targets':indices,'frame_count':len(rows),'prompt_sha256':record['prompt_sha256'],
          'prompt_mask_sha256':prompt['refined_mask_sha256'],'frames_metadata_sha256':record['frames_sha256'],
          'target_image_sha256':[rows[i]['image_sha256'] for i in indices]}
    for group1,group2 in (('train','validation'),('train','protected_priority'),('validation','protected_priority')):
        for aseq in groups[group1]:
            for bseq in groups[group2]:
                if hash_sets[aseq]&hash_sets[bseq]:raise ValueError(f'Exact image overlap between {aseq} and {bseq}')
    manifest={'groups':groups,'sequences':report,'selection':'Five evenly spaced indices, independent of labels.',
      'limitation':'Patient identity unknown; sequence separation and exact-image hash checks do not establish patient independence.',
      'protocol_sha256':digest(PROTOCOL),'data_manifest_sha256':digest(ROOT/'data/polypgen_sequence.jsonl'),
      'merged_results_sha256':digest(a.merged/'results.json'),'source_sha256':digest(Path(__file__))}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'manifest_sha256':digest(out/'manifest.json')},indent=2)+'\n')
    print(json.dumps({'passed':True,'targets':{k:v['targets'] for k,v in report.items()}},indent=2))
if __name__=='__main__':main()
