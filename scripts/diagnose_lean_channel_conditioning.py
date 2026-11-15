#!/usr/bin/env python3
"""Post-hoc, label-free sensitivity of one frozen gate to prompt descriptors."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_channel_feature_capture import sha
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('dataset','training','training-audit','validation-verification','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();d=read(a.dataset/'dataset.json');t=read(a.training/'training_receipt.json');ta=read(a.training_audit/'verification.json');va=read(a.validation_verification/'verification.json')
    if not ta['passed'] or ta['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_repeat.py') or ta['first_receipt_sha256']!=digest(a.training/'training_receipt.json') or not va['passed'] or va['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_validation.py') or not t['passed']:
        raise ValueError('Training or validation audit changed')
    if d['groups']['validation']!=['seq9','seq10'] or digest(a.dataset/'dataset.json')!=ta['dataset_sha256']:
        raise ValueError('Validation scope changed')
    path=a.training/'prompt_conditioned.pt'
    if digest(path)!=t['arms']['prompt_conditioned']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    model=PromptContrastChannelAttention(checkpoint=path,expected_backbone_sha256=d['backbone_sha256'],expected_feature_pipeline_sha256=d['feature_pipeline_sha256']).eval().requires_grad_(False)
    samples={}
    for seq in ('seq9','seq10'):
        item=min((x for x in d['samples'] if x['group']=='validation' and x['sequence']==seq),key=lambda x:x['frame'])
        feature=Path(item['feature_path'])
        if digest(feature)!=item['feature_file_sha256']:raise ValueError('Feature changed')
        samples[seq]=torch.load(feature,map_location='cpu',weights_only=True)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);rows=[]
    with torch.inference_mode():
        for seq,other in (('seq9','seq10'),('seq10','seq9')):
            v=samples[seq];w=samples[other];fg,bg=v['foreground'],v['background'];zeros=torch.zeros_like(fg)
            variants={'own':(fg,bg),'other_sequence':(w['foreground'],w['background']),'zero':(zeros,zeros)};scores={}
            for name,(gfg,gbg) in variants.items():
                score,weights=model(v['feature'],fg,bg,gate_foreground=gfg,gate_background=gbg)
                scores[name]={'score':score.cpu(),'weights':weights.cpu()}
            base=scores['own'];row={'sequence':seq,'frame':v['frame'],'feature_sha256':sha(v['feature']),'comparisons':{}}
            for name in ('other_sequence','zero'):
                row['comparisons'][name]={'mean_abs_channel_weight_change':float((scores[name]['weights']-base['weights']).abs().mean()),
                  'max_abs_channel_weight_change':float((scores[name]['weights']-base['weights']).abs().max()),
                  'mean_abs_score_change':float((scores[name]['score']-base['score']).abs().mean()),
                  'max_abs_score_change':float((scores[name]['score']-base['score']).abs().max())}
            rows.append(row)
    result={'passed':True,'scope':'Post-hoc, label-free fixed-gate descriptor perturbation; not a causal training or video test.',
      'rows':rows,'checkpoint_sha256':digest(path),'dataset_sha256':digest(a.dataset/'dataset.json'),
      'training_verification_sha256':digest(a.training_audit/'verification.json'),'validation_verification_sha256':digest(a.validation_verification/'verification.json'),
      'adapter_sha256':digest(ROOT/'modeling/lean_channel_attention.py'),'runner_sha256':digest(Path(__file__))}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
