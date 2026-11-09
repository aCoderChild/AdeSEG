"""Reuse the native indexed DAVIS fixture and existing multi-object rollout gate."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from datasets.video import ManifestFrame
from modeling.rde_value_repair import MemoryValueRepair,bind_native_reads
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,PROTOCOL,_guard_sources,_model_digest
from scripts.verify_token_conditioned_key_adapter import davis_frames
from scripts.verify_mavos_focal_value_modulation import multi_object_rollout


def run(reference,out):
    complete=json.loads((reference/'completion.json').read_text())
    if not complete['passed'] or complete['results_sha256']!=digest(reference/'results.json'):raise RuntimeError('Reference engineering gate differs')
    identity=json.loads((reference/'identity.json').read_text());_guard_sources(identity['sources'])
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261109)
    images,labels=davis_frames('bike-packing',5)
    frames=[ManifestFrame('bike-packing',i,image,label) for i,(image,label) in enumerate(zip(images,labels))]
    indexed=np.asarray(Image.open(labels[0]));masks={i:torch.from_numpy((indexed==i).copy()) for i in [1,2]}
    if any(not mask.any() for mask in masks.values()):raise RuntimeError('Persistent indexed prompt IDs missing')
    repo,revision=activate_author_repo();base=json.loads(PROTOCOL.read_text())
    if revision!=base['model']['sam2_revision'] or digest(CHECKPOINT_DEFAULT)!=identity['checkpoint_sha256']:raise RuntimeError('Native model identity differs')
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=base['model']['configuration']),repo)
    model.requires_grad_(False).eval();core=_model_digest(model)
    def rollout(order,adapter):
        with bind_native_reads(model,adapter) if adapter is not None else nullcontext():
            result=multi_object_rollout(model,frames,0,out,order,masks,None if adapter is None else [adapter]*4)
        if adapter is not None:
            if {r['object_id'] for r in result['reads']}!=set(order):raise RuntimeError('Read ownership differs from persistent IDs')
            if any(r['layers']!=[0,1,2,3] or not r['native_positions_verified'] or not r['stored_mask_alignment'] for r in result['reads']):
                raise RuntimeError('Memory alignment/position/layer gate failed')
            if len(result['value_calls'])!=len(result['reads'])*4 or any(not r['pointer_tail_bit_identical'] for r in result['value_calls']):
                raise RuntimeError('Multi-object pointer isolation failed')
        return result
    native=rollout([1,2],None);zero=rollout([1,2],MemoryValueRepair())
    if native['outputs']!=zero['outputs']:raise RuntimeError('Multi-object zero/native parity failed')
    path=reference/'seq2_segmentation_plus_consistency_resume.pt'
    expected=json.loads((reference/'results.json').read_text())['sequences']['seq2']['arms']['segmentation_plus_consistency']['checkpoint_sha256']
    if digest(path)!=expected:raise RuntimeError('Nonzero trained fixture checkpoint differs')
    adapter=MemoryValueRepair();adapter.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['adapter'])
    if adapter.mlp.layers[-1].weight.count_nonzero()==0:raise RuntimeError('Trained nonzero fixture is still identity')
    trained=rollout([1,2],adapter);reverse=rollout([2,1],adapter)
    if trained['outputs']!=reverse['outputs']:raise RuntimeError('Persistent-ID prompt order invariance failed')
    singles={}
    for object_id in [1,2]:
        singles[object_id]=rollout([object_id],adapter)
        if singles[object_id]['outputs']!={k:v for k,v in trained['outputs'].items() if k.endswith(f':{object_id}')}:
            raise RuntimeError('Multi-object output differs from isolated object')
    if _model_digest(model)!=core:raise RuntimeError('SAM2 parameters changed')
    _guard_sources(identity['sources'])
    result={'passed':True,'scope':'DAVIS bike-packing natural indexed IDs1/2, outputs frames0-2 only; engineering, not benchmark efficacy',
        'zero_native_bit_exact':True,'nonzero_prompt_order_invariant':True,'nonzero_single_object_isolation_exact':True,
        'reference_results_sha256':digest(reference/'results.json'),'adapter_checkpoint_sha256':digest(path),
        'script_sha256':digest(Path(__file__)),'input_files':[{'image':digest(i),'indexed_mask':digest(m)} for i,m in zip(images,labels)],
        'native':native,'zero':zero,'trained':trained,'reverse':reverse,'singles':singles,'efficacy_authorized':False}
    atomic_json(out/'results.json',result);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'efficacy_authorized':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    try:run(a.reference,a.output_dir)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
