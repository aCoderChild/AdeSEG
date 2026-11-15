#!/usr/bin/env python3
"""Independently verify frozen, repeated cross-frame pool engineering runs."""
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def require(condition, message):
    if not condition: raise ValueError(message)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pool-root',type=Path,required=True)
    p.add_argument('--raw-gate',type=Path,required=True)
    p.add_argument('--old-pool-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); base=a.pool_root
    identity=None; summary={}
    raw=read(a.raw_gate/'records.json');raw_events=read(a.raw_gate/'token_window_events.json')
    raw_pos=read(a.raw_gate/'positions.json')
    for strategy in ('score','distributed','usage','contrast','three_region'):
        replays=[]
        for attempt in (1,2):
            folder=base/f'{strategy}_gate_attempt0{attempt}'
            receipt=read(folder/'pool_observation.json');record=read(folder/'records.json')
            events=read(folder/'pool_events.json');scores=read(folder/'source_scores.json')
            require(receipt['passed'] and receipt['sequence']=='seq3' and receipt['mode']=='gate' and receipt['strategy']==('score_only' if strategy=='score' else strategy),'invalid receipt')
            if identity is None: identity=receipt['identity']
            require(receipt['identity']==identity,'identities differ')
            for key,name in (('records_sha256','records.json'),('events_sha256','pool_events.json'),('source_scores_sha256','source_scores.json')):
                require(receipt[key]==digest(folder/name),f'{folder}: {name} hash mismatch')
            require(receipt['source_score_events']==len(scores) and bool(scores),'missing source scores')
            require(read(folder/'completion.json')=={'passed':True,'records_sha256':digest(folder/'records.json')},'completion mismatch')
            integrity=read(folder/'parameter_integrity_after.json')
            require(integrity['unchanged'] and not integrity['changed_names'] and integrity['before']==integrity['after'],'model parameters changed')
            for key,name in (('positions_sha256','positions.json'),('selections_sha256','selections.json'),('memory_events_sha256','memory_events.json')):
                require(record[key]==digest(folder/name),f'{name} changed')
            require(record['mask_sha256']==[digest(folder/f'{i:06d}.png') for i in range(record['frames'])],'mask hashes changed')
            pos=read(folder/'positions.json')
            require([x['pointer_keys'] for x in pos]==[x['pointer_keys'] for x in raw_pos],'pointer count differs')
            if strategy=='score': require(record==raw and events==raw_events,'score capture changed raw-window run')
            else:
                plans=[e for e in events if e['stage']=='admission_plan'];writes=[e for e in events if e['stage']=='after_write']
                require(len(plans)==len(writes)>0,'missing pool events')
                for plan,write in zip(plans,writes):
                    kept=sum(plan['pool_counts'].values())
                    require(plan['frame']==write['frame'] and plan['condition_tokens']==plan['new_tokens']==4096 and plan['insurance_reserve']==3072 and plan['pool_budget']==5120,'protected budget mismatch')
                    require(kept==plan['pool_kept']==min(5120,plan['pool_candidates']) and plan['planned_reserved_tokens']==write['spatial_tokens']+3072<=16384,'pool underfilled or budget exceeded')
                    require(set(plan['pool_counts'])==set(plan['pool_ids_sha256']),'missing token ID hashes')
                summary[strategy]={'max_history_sources':max(len(e['pool_counts']) for e in plans),'max_auxiliary_metadata_bytes':max(e['score_metadata_bytes'] for e in plans),'admissions':len(plans)}
                if strategy=='distributed': require(summary[strategy]['max_history_sources']>5,'history did not exceed five sources')
            if strategy in ('distributed','usage','contrast'):
                old=a.old_pool_root/f'{strategy}_gate_attempt03'
                require(record==read(old/'records.json') and pos==read(old/'positions.json') and read(folder/'selections.json')==read(old/'selections.json') and read(folder/'memory_events.json')==read(old/'memory_events.json'),'boundary observer changed existing control')
            replays.append((record,events,scores,pos,read(folder/'selections.json'),read(folder/'memory_events.json')))
        require(replays[0]==replays[1],f'{strategy} independent replays differ')
    require(identity['runner_sha256']==digest(ROOT/'scripts/run_lean_region_token_pool.py'),'runner changed')
    require(identity['protocol_sha256']==digest(ROOT/'protocols/lean_region_token_pool_v1.json'),'protocol changed')
    for name,sha in identity['sources'].items(): require(digest(ROOT/name)==sha,f'{name} changed')
    result={'passed':True,'identity':identity,'summary':summary,'verifier_sha256':digest(Path(__file__)),'interpretation':'Engineering replays only; no evidence of segmentation benefit or token utility.'}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
