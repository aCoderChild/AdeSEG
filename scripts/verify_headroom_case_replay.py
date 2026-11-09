"""Fresh-process replay of the first frozen TRAIN oracle case, not efficacy.

Reuse the same source teacher, read-only query and author propagation. Compare
immutable saved output/score/key/position/pointer receipts, excluding wall time.
There is no optimizer or model training to resume in this diagnostic.
"""
import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.complementary_memory.training import capture_current_features,clone_tensors
from modeling.native_memory_layout import selected_memory_frames
from scripts.mask_memory_teacher import capture_native_mask_inputs,same_frame_teachers
from scripts.memory_value_query import MEMORY_FIELDS,query,spatial_values
from scripts.mavos_value_utils import activate_author_repo,atomic_json,prepare
from scripts.memory_token_io import digest,validate_output
from scripts.prepare_mask_memory_headroom import DESIGN
from scripts.run_author_baseline import build_author_predictor
from scripts.run_memory_value_headroom import load_case,score
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,PROTOCOL,_guard_sources,_model_digest,_tree_digest


def run(reference,out):
    identity=json.loads((reference/'identity.json').read_text());_guard_sources(identity['sources'])
    plan=json.loads((reference/'case_plan_frozen.json').read_text());case=plan['cases'][0]
    if case['id']!='seq2_t7':raise RuntimeError('First frozen case differs')
    expected=load_case(reference,case['id'],digest(reference/'identity.json'))
    if expected is None:raise RuntimeError('First case not complete')
    design=json.loads(DESIGN.read_text());protocol=json.loads(PROTOCOL.read_text())
    if digest(CHECKPOINT_DEFAULT)!=identity['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    sources=dict(identity['sources']);sources['scripts/verify_headroom_case_replay.py']=digest(Path(__file__))
    atomic_json(out/'identity.json',{'reference_identity_sha256':digest(reference/'identity.json'),
        'case_result_sha256':digest(reference/'cases/seq2_t7.json'),'sources':sources,
        'threads':1,'torch_version':torch.__version__,'torch_build':torch.__config__.show()})
    if torch.__version__!=identity['torch_version'] or torch.__config__.show()!=identity['torch_build']:
        raise RuntimeError('Fresh replay runtime differs')
    for relative in sources:
        p=out/'source'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/relative).read_bytes())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(design['seed'])
    repo,revision=activate_author_repo()
    if revision!=protocol['model']['sam2_revision']:raise RuntimeError('Native revision differs')
    all_frames=load_manifest(MANIFEST,'dev')['seq2'];frames=all_frames[plan['sequences']['seq2']['prompt_manifest_offset']:]
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,
        model_config=protocol['model']['configuration']),repo);model.requires_grad_(False).eval();core=_model_digest(model)
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        state,prompt=prepare(model,frames,Path(temporary))
        if prompt!=0:raise RuntimeError('Prompt differs')
        masks=OrderedDict();iterator=model.propagate_in_video(state,start_frame_idx=0)
        try:
            while True:
                with capture_native_mask_inputs(model) as cap,torch.inference_mode():
                    frame,ids,_=next(iterator)
                if ids!=[1] or cap['count']!=1:raise RuntimeError('Native capture ownership differs')
                masks[frame]=clone_tensors(cap['latest'])
                if frame==7:break
        finally:iterator.close()
        native_bank=state['output_dict_per_obj'][0]
        target={f:native_bank['non_cond_frame_outputs'][7][f] for f in MEMORY_FIELDS}
        if _tree_digest(target)!=expected['native_propagation_sha256']:raise RuntimeError('Fresh native prefix differs')
        bank=clone_tensors({'cond_frame_outputs':native_bank['cond_frame_outputs'],
            'non_cond_frame_outputs':{k:v for k,v in native_bank['non_cond_frame_outputs'].items() if k<7}})
        if _tree_digest(bank)!=expected['native_bank_sha256']:raise RuntimeError('Fresh native past bank differs')
        spans=selected_memory_frames(model,7,bank,len(frames));values=spatial_values(bank,spans)
        features=capture_current_features(model,state,7);teachers={}
        for span in spans:
            source=span['frame']
            if source in bank['cond_frame_outputs']:continue
            visual=capture_current_features(model,state,source)
            truth=torch.from_numpy((load_label_mask(frames[source].mask_path,[1])==1).copy())[None,None]
            teachers[source]=same_frame_teachers(model,visual,bank['non_cond_frame_outputs'][source],truth,masks[source])
        truth=load_label_mask(frames[7].mask_path,[1])==1
        checks={}
        for arm in ('native','identity_values','mask_only','mask_and_presence'):
            current=query(model,features,bank,7,len(frames)) if arm=='native' else query(
                model,features,bank,7,len(frames),values,spatial_values(bank,spans,teachers,arm))
            scored=score(current['output'],truth)
            for key in ('output_sha256','keys','calls','values','value_changes'):
                if current[key]!=expected['arms'][arm][key]:raise RuntimeError(f'Fresh {arm} {key} differs')
            for key in ('scores','prediction_nonempty','object_score'):
                if scored[key]!=expected['arms'][arm][key]:raise RuntimeError(f'Fresh {arm} {key} differs')
            checks[arm]={'output_sha256':current['output_sha256'],'all_receipt_fields_exact':True}
        if _tree_digest(bank)!=expected['native_bank_sha256']:raise RuntimeError('Replay changed bank')
        model.reset_state(state)
    if _model_digest(model)!=core:raise RuntimeError('Replay changed SAM2 weights')
    _guard_sources(sources)
    atomic_json(out/'results.json',{'case':case['id'],'arms':checks,'passed':True,
        'scope':'One fixed TRAIN case in fresh process; no optimizer/training resume claim',
        'efficacy_authorized':False})
    atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),
        'efficacy_authorized':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args();existed=a.output_dir.exists()
    try:run(a.reference,a.output_dir)
    except Exception as error:
        if not existed and a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise
