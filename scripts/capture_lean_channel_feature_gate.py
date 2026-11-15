#!/usr/bin/env python3
"""Label-free no-op feature capture on the pinned seq3 raw-window gate."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_feature_capture import capture_channel_features
from scripts.memory_token_io import digest
import scripts.run_lean_cross_frame_pool as runner


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(add_help=False)
    for name in ('preflight','capture-reference','output'):p.add_argument('--'+name,type=Path,required=True)
    a,rest=p.parse_known_args();manifest=read(a.preflight/'manifest.json');pre=read(a.preflight/'completion.json')
    if not pre['passed'] or pre['manifest_sha256']!=digest(a.preflight/'manifest.json') or manifest['protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_training_pilot_v1.json') or manifest['source_sha256']!=digest(ROOT/'scripts/preflight_lean_channel_training.py'):
        raise ValueError('Training preflight changed')
    if '--strategy' not in rest or rest[rest.index('--strategy')+1]!='score_only' or '--sequence' not in rest or rest[rest.index('--sequence')+1]!='seq3' or '--mode' not in rest or rest[rest.index('--mode')+1]!='gate':
        raise ValueError('Feature engineering gate requires score-only seq3')
    reference=read(a.capture_reference/'pool_observation.json')
    if not reference['passed'] or reference['strategy']!='score_only' or digest(a.capture_reference/'records.json')!=reference['records_sha256'] or digest(a.capture_reference/'pool_events.json')!=reference['events_sha256']:
        raise ValueError('Reference score-only gate changed')
    targets=set(manifest['sequences']['seq3']['targets']);events=[];original=runner.capture_pool_scores
    @contextmanager
    def capture(model,observations):
        with original(model,observations):
            with capture_channel_features(model,targets,a.output,events):yield
    runner.capture_pool_scores=capture
    try:
        sys.argv=['run_lean_cross_frame_pool.py',*rest,'--output',str(a.output)];runner.main()
    finally:runner.capture_pool_scores=original
    for name in ('records.json','pool_events.json','source_scores.json','positions.json','selections.json','memory_events.json'):
        if digest(a.capture_reference/name)!=digest(a.output/name):raise ValueError('Feature observer changed baseline: '+name)
    if {e['frame'] for e in events}!=targets:raise ValueError('Capture targets changed')
    for e in events:e['file_sha256']=digest(a.output/'features'/f"{e['frame']:06d}.pt")
    result={'passed':True,'sequence':'seq3','targets':sorted(targets),'features':events,'preflight_sha256':digest(a.preflight/'manifest.json'),
      'reference_receipt_sha256':digest(a.capture_reference/'pool_observation.json'),'output_receipt_sha256':digest(a.output/'pool_observation.json'),
      'capture_sha256':digest(ROOT/'modeling/lean_channel_feature_capture.py'),'runner_sha256':digest(Path(__file__))}
    (a.output/'feature_capture_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
