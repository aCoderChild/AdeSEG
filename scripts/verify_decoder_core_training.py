"""Engineering-only ordinary decoder training on verified official native banks."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.decoder_adaptation import train_decoder_core
from modeling.decoder_core_training import decoder_clip_loss
from modeling.native_history_cache import capture_cached_block, load_bank
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo
from scripts.run_regional_refinement import save_tensor, write
from training.tracking import start_wandb

PROTOCOL='protocols/decoder_core_training_gate_v1.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-gate',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--wandb-mode',choices=['offline','disabled'],default='offline')
    args=parser.parse_args()
    out=validate_output(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((ROOT/PROTOCOL).read_text())
    gate=json.loads((args.native_gate/'completion.json').read_text())
    audit=json.loads((args.native_gate/'independent_audit.json').read_text())
    if not gate.get('passed') or gate['scope']!='gate' or not audit.get('passed'):
        raise RuntimeError('Official native gate and independent audit required')
    if audit.get('scope')!='gate' or set(audit['sequences'])!=set(gate['entries']):
        raise RuntimeError('Independent audit covers different native sequences')
    for sequence,folder_name in gate['entries'].items():
        folder=Path(folder_name)
        record=audit['sequences'][sequence]
        if (digest(folder/'receipt.json')!=record['receipt_sha256'] or
                digest(folder/'completion.json')!=record['completion_sha256']):
            raise RuntimeError('Independently audited native evidence changed')
    setup=json.loads((args.native_gate/'setup.json').read_text())
    sources={**gate['sources'],**{p:digest(ROOT/p) for p in
        [PROTOCOL,'scripts/verify_decoder_core_training.py','modeling/decoder_core_training.py',
         'tests/test_decoder_core_training.py']}}
    if any(digest(ROOT/p)!=h for p,h in sources.items()):
        raise RuntimeError('Frozen source changed')
    parent=json.loads((ROOT/'protocols/encoder_decoder_bottleneck_v1.json').read_text())
    if digest(Path(parent['checkpoint']))!=setup['checkpoint_sha256']:
        raise RuntimeError('Official checkpoint differs')
    write(out/'setup.json',{'sources':sources,'protocol':protocol,
        'native_gate_sha256':digest(args.native_gate/'completion.json'),
        'native_audit_sha256':digest(args.native_gate/'independent_audit.json')})
    for key in ['WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR','TMPDIR']:
        folder=out/key.lower();folder.mkdir()
        os.environ[key]=str(folder)
    run=start_wandb('ade-seg','adenoid-hypertrophy',out.name,args.wandb_mode,None,
        protocol,out,['ordinary-decoder','engineering-only'])
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    repo,_=activate_author_repo()
    model=build_author_predictor(SimpleNamespace(method='sam2',device='cpu',
        checkpoint=Path(parent['checkpoint']),model_config=parent['model_config']),repo)
    selected=train_decoder_core(model)
    if sum(p.numel() for name,p in selected)!=4190437:
        raise RuntimeError('Decoder-core parameter scope differs')
    frozen=lambda:{n:p for n,p in model.named_parameters() if not p.requires_grad}
    frozen_sha=fingerprint(frozen())['sha256']
    core_sha=fingerprint({n:p for n,p in selected})['sha256']
    videos={**load_manifest(ROOT/parent['manifest'],'dev'),**load_manifest(ROOT/parent['manifest'],'test')}
    blocks,metadata=[],[]
    for spec in protocol['clips']:
        frames=videos[spec['sequence']]
        actual=[r.frame_index for r in sample_clip(frames,8,spec['seed'])]
        if actual!=spec['frames']:
            raise RuntimeError('Fixed author sample differs')
        folder=Path(gate['entries'][spec['sequence']])
        receipt=json.loads((folder/'receipt.json').read_text())
        payload=load_bank(folder/'bank.pt',receipt,setup['checkpoint_sha256'],
            setup['manifest_sha256'],setup['model_config'])
        samples,record=capture_cached_block(model,frames,actual,out,payload)
        blocks.append(samples);metadata.append(record)
    prefixes_sha=fingerprint([[s['output_dict'] for s in block] for block in blocks])['sha256']
    before=[]
    for samples in blocks:
        with torch.no_grad():
            value,_=decoder_clip_loss(model,samples,check_native=True)
        before.append(float(value))
    value,info=decoder_clip_loss(model,blocks[0],check_native=True,retain_first_write=True)
    gradient=torch.autograd.grad(info['last_loss'],info['first_write'],retain_graph=True)[0]
    if not torch.isfinite(gradient).all() or not gradient.abs().sum()>0:
        raise RuntimeError('Later decoder loss does not reach the earlier predicted memory')
    write(out/'replay.json',{'passed':True,'initial_native_losses':before,
        'last_loss_to_first_memory_gradient_l1':float(gradient.abs().sum()),'blocks':metadata})
    del value,info,gradient
    optimizer=torch.optim.AdamW([p for n,p in selected],lr=5e-6,weight_decay=.01)
    history=[]
    for step in range(protocol['updates']):
        optimizer.zero_grad(set_to_none=True)
        loss,info=decoder_clip_loss(model,blocks[step%len(blocks)])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite decoder training loss')
        loss.backward()
        active={n:float(p.grad.abs().sum()) for n,p in selected if p.grad is not None}
        if (not active or sum(active.values())<=0 or
                any(not torch.isfinite(p.grad).all() for n,p in selected if p.grad is not None)):
            raise RuntimeError('Missing or nonfinite decoder gradients')
        norm=torch.nn.utils.clip_grad_norm_([p for n,p in selected],1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite total decoder gradient norm')
        optimizer.step()
        if fingerprint(frozen())['sha256']!=frozen_sha:
            raise RuntimeError('Optimizer changed frozen native parameters')
        row={'update':step+1,'clip':step%len(blocks),'loss':float(loss.detach()),
             'gradient_norm':float(norm),'active_parameter_tensors':len(active),'gradient_l1':active}
        history.append(row);write(out/'training.json',history)
        run.log({'train/loss':row['loss'],'train/gradient_norm':row['gradient_norm']},step=step+1)
        print('verified decoder update',step+1,flush=True)
        del loss,info
    with torch.no_grad():
        after=[float(decoder_clip_loss(model,samples)[0]) for samples in blocks]
    if not all(math.isfinite(value) for value in before+after):
        raise RuntimeError('Nonfinite replay loss')
    if fingerprint([[s['output_dict'] for s in block] for block in blocks])['sha256']!=prefixes_sha:
        raise RuntimeError('Training modified a native teacher prefix')
    weights={n:p.detach().clone() for n,p in selected}
    if fingerprint(weights)['sha256']==core_sha or any(not torch.isfinite(p).all() for p in weights.values()):
        raise RuntimeError('Decoder core did not update finitely')
    save_tensor(weights,out/'decoder_core.pt')
    write(out/'completion.json',{'passed':True,'engineering_only':True,'updates':len(history),
        'sources':sources,'trainable_parameters':4190437,'loss_before':before,'loss_after':after,
        'frozen_parameters_unchanged':True,'native_prefixes_unchanged':True,
        'weights_sha256':digest(out/'decoder_core.pt'),
        'note':'No validation efficacy measured. Diagnostic weights never initialise CV.'})
    run.summary.update({'engineering_passed':True,'updates':len(history)})
    run.finish()


if __name__=='__main__':
    main()
