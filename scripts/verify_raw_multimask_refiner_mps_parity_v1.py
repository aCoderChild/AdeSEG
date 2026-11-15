#!/usr/bin/env python3
"""CPU/MPS value and gradient parity on frozen raw-candidate training reads.

Only train labels are opened. This check does not train a candidate, change
inference, or authorize video propagation.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.raw_multimask_refiner import RawMultimaskRefiner
from modeling.robust_segmentation_loss import robust_bce_dice_loss
from scripts.memory_token_io import digest,validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import load_target,read_capture,to_video
from scripts.train_region_mixed_loss_v1 import mixed_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-preflight','raw-audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise RuntimeError('MPS with fast math and fallback disabled is required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest,mhash=mixed_manifest(a.data_preflight);rawpath=a.raw_audit/'manifest.json'
    raw=json.loads(rawpath.read_text());done=json.loads((a.raw_audit/'completion.json').read_text())
    if not done['passed'] or done['manifest_sha256']!=digest(rawpath) or not raw['passed'] or raw['manifest_sha256']!=mhash or raw['validation_labels_opened'] is not False or raw['runner_sha256']!=digest(ROOT/'scripts/audit_raw_multimask_mixed_v1.py'):
        raise ValueError('Frozen raw candidate audit differs')
    sample={(x['sequence'],x['frame']):x for x in manifest['samples'] if x['split']=='train'}
    raws={(x['sequence'],x['frame']):x for x in raw['rows'] if x['split']=='train'}
    if len(sample)!=39 or set(sample)!=set(raws):raise ValueError('Training raw-candidate coverage differs')
    torch.manual_seed(20261114);cpu=RawMultimaskRefiner().train()
    # Nonzero fixture weights exercise gradients through raw and feature paths.
    torch.nn.init.normal_(cpu.head[-1].weight,mean=0,std=0.01)
    torch.nn.init.normal_(cpu.raw_adapter.weight,mean=0,std=0.01)
    gpu=RawMultimaskRefiner().train();gpu.load_state_dict(cpu.state_dict(),strict=True);gpu.to('mps')
    checks=[]
    for key in (('seq4',24),('seq4',21),('seq6',90),('seq13',62)):
        item=sample[key];entry=raws[key]
        if digest(entry['raw_file'])!=entry['raw_sha256'] or entry['snapshot_sha256']!=item['snapshot_sha256']:
            raise ValueError('Frozen candidate or snapshot changed')
        blob=torch.load(entry['raw_file'],map_location='cpu',weights_only=True)
        early,deep,logits,shape=read_capture(item);target=load_target(item,shape)
        inputs=(early,deep,logits,blob['raw_candidates'],blob['predicted_ious'])
        for kind in ('bce','robust_bce'):
            values=[]
            for model,device in ((cpu,'cpu'),(gpu,'mps')):
                model.zero_grad(set_to_none=True)
                result=model(*(x.to(device) for x in inputs));prediction=to_video(result,shape);truth=target.to(device)
                if kind=='robust_bce':loss=robust_bce_dice_loss(prediction,truth)
                else:
                    bce=torch.nn.functional.binary_cross_entropy_with_logits(prediction,truth)
                    prob=prediction.sigmoid();loss=bce+1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
                if not torch.isfinite(loss) or not torch.isfinite(result).all():raise ValueError('Nonfinite forward or loss')
                loss.backward();grads=[None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()]
                if any(g is None or not torch.isfinite(g).all() for g in grads):raise ValueError('Nonfinite or missing gradient')
                values.append((result.detach().cpu(),loss.detach().cpu(),grads))
            torch.testing.assert_close(values[0][0],values[1][0],atol=1e-3,rtol=1e-3)
            torch.testing.assert_close(values[0][1],values[1][1],atol=1e-3,rtol=1e-3)
            maximum=diff=0.0
            for left,right in zip(values[0][2],values[1][2],strict=True):
                torch.testing.assert_close(left,right,atol=1e-3,rtol=1e-3)
                maximum=max(maximum,float(left.abs().max()));diff=max(diff,float((left-right).abs().max()))
            checks.append({'sequence':key[0],'frame':key[1],'capture_device':item['capture_device'],'loss':kind,
                           'cpu_loss':float(values[0][1]),'mps_loss':float(values[1][1]),
                           'maximum_output_difference':float((values[0][0]-values[1][0]).abs().max()),
                           'maximum_gradient_magnitude':maximum,'maximum_gradient_difference':diff})
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'scope':'Fixed-read train-only parity; no validation labels or video inference.',
            'manifest_sha256':mhash,'raw_manifest_sha256':digest(rawpath),'model_sha256':digest(ROOT/'modeling/raw_multimask_refiner.py'),
            'conditioned_source_sha256':digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
            'base_source_sha256':digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
            'robust_loss_sha256':digest(ROOT/'modeling/robust_segmentation_loss.py'),
            'runner_sha256':digest(Path(__file__)),'torch':str(torch.__version__),
            'mps_fallback':'0','mps_fast_math':'0','deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'checks':checks}
    (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
