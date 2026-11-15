#!/usr/bin/env python3
"""Label-free real-feature optimization/serialization engineering check."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_channel_training import build_trainable_gate,train_gate,parameter_hash
from modeling.lean_channel_feature_capture import sha
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('feature-gate','feature-reference','optimization-protocol','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();g=read(a.feature_gate/'verification.json');ref=read(a.feature_reference/'channel_feature_receipt.json')
    if not g['passed'] or g['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_pipeline_gates.py') or g['runner_sha256']!=digest(ROOT/'scripts/capture_lean_channel_features.py') or g['feature_pipeline_sha256']!=ref['feature_pipeline_sha256'] or not ref['passed']:
        raise ValueError('Real-feature engineering input invalid')
    item=next(x for x in ref['features'] if x['frame']==7);path=a.feature_reference/'features/000007.pt'
    if digest(path)!=item['file_sha256']:raise ValueError('Engineering feature changed')
    sample=torch.load(path,map_location='cpu',weights_only=True)
    for k in ('feature','foreground','background'):
        if sha(sample[k])!=item[k+'_sha256']:raise ValueError('Engineering tensor changed')
    # Synthetic occupancy is only an engineering fixture; no ground truth used.
    sample.update(sequence='synthetic',target=torch.linspace(0,1,4096));foreign={'synthetic':(sample['foreground'].flip(0),sample['background'].flip(0))}
    cfg=read(a.optimization_protocol)['training'];torch.set_num_threads(2);out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);arms={}
    for arm in ('prompt_conditioned','descriptor_independent','mismatched_prompt'):
        model=build_trainable_gate(cfg['seed']);r=train_gate(model,[sample],arm,foreign,1,cfg['learning_rate'],cfg['weight_decay'])
        fixture={'format':'sam2_early_prompt_cbam_v1','channels':32,'author_sha256':author_sha256(),'training_steps':r['steps'],
          'feature_interface':'sam2.1_hiera_base_plus.conv_s0.fpn0','backbone_sha256':ref['checkpoint_sha256'],'feature_pipeline_sha256':g['feature_pipeline_sha256'],
          'train_manifest_sha256':'0'*64,'gate_state_dict':{k:v.detach().cpu().clone() for k,v in model.gate.state_dict().items()},'engineering_only':True}
        path=out/f'{arm}_fixture.pt';torch.save(fixture,path)
        try:
            PromptContrastChannelAttention(checkpoint=path,expected_backbone_sha256=ref['checkpoint_sha256'],expected_feature_pipeline_sha256=g['feature_pipeline_sha256'])
            raise ValueError('Engineering fixture was accepted for inference')
        except ValueError as error:
            if 'Engineering fixture cannot' not in str(error):raise
        # Reload only the raw gate parameters into an explicitly training-mode
        # model. The synthetic target never creates an inference checkpoint.
        loaded=build_trainable_gate(cfg['seed']);loaded.gate.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['gate_state_dict'],strict=True)
        score,weights=loaded(sample['feature'],sample['foreground'],sample['background'])
        if not torch.isfinite(score).all() or parameter_hash(loaded)!=r['final_parameter_sha256']:raise ValueError('Trained fixture did not reload')
        arms[arm]={'initial_parameter_sha256':r['initial_parameter_sha256'],'final_parameter_sha256':r['final_parameter_sha256'],'steps':r['steps'],
          'loss':r['epoch_losses'][0],'score_sha256':sha(score),'weights_sha256':sha(weights),'fixture_sha256':digest(path)}
    if len({r['initial_parameter_sha256'] for r in arms.values()})!=1:raise ValueError('Initialization mismatch')
    result={'passed':True,'scope':'Synthetic target with one real feature; engineering only and no usable trained checkpoint.',
      'arms':arms,'feature_sha256':item['feature_sha256'],'optimization_protocol_sha256':digest(a.optimization_protocol),
      'adapter_sha256':digest(ROOT/'modeling/lean_channel_attention.py'),'training_sha256':digest(ROOT/'modeling/lean_channel_training.py'),
      'loss_sha256':digest(ROOT/'modeling/lean_channel_training_loss.py'),'runner_sha256':digest(Path(__file__))}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'arms':list(arms)}))
if __name__=='__main__':main()
