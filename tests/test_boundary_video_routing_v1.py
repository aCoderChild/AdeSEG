import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from modeling.lean_boundary_refinement_routing_v1 import route_boundary_refinement, _stats
from test_lean_contrast_refinement_integration import Predictor, Refiner, run
from scripts import run_boundary_refiner_priority_v1 as runner
from scripts import verify_boundary_video_short_gate_v1 as gate


class NumericPredictor(Predictor):
    def _track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                    output_dict, num_frames, track_in_reverse, prev_sam_mask_logits,
                    inference_state=None, obj_idx=None):
        current, sam, early, deep = super()._track_step(
            frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
            feat_sizes, point_inputs, mask_inputs, output_dict, num_frames, track_in_reverse,
            prev_sam_mask_logits, inference_state, obj_idx)
        return current, (*sam[:-1], torch.tensor([[-2.]])), early, deep
    def _encode_memory_in_output(self, current_vision_feats, feat_sizes, point_inputs,
                                 run_mem_encoder, high_res_masks, object_score_logits,
                                 current_out):
        assert object_score_logits.item() == -2
        self.last_memory = high_res_masks.clone() if run_mem_encoder else None
        current_out['memory'] = self.last_memory


@pytest.mark.parametrize('arm,out_delta,mem_delta', [('A',0,0),('B',1,0),('C',0,1),('D',1,1)])
def test_observed_routes_preserve_native_state(arm,out_delta,mem_delta):
    model=NumericPredictor().eval(); original, old=run(model); trace=[]
    with route_boundary_refinement(model, Refiner().eval(), arm, test_only=True, telemetry=trace):
        compact,out=run(model)
        assert torch.equal(compact['pred_masks'],original['pred_masks'])
        assert torch.equal(out,old+out_delta)
        assert torch.allclose(compact['memory'],original['memory']+mem_delta)
        conditioned,cout=run(model,conditioning=True)
        assert torch.equal(cout,old)
        assert torch.equal(conditioned['memory'],original['memory'])
    assert len(trace)==2 and trace[0]['conditioning'] is False and trace[1]['conditioning'] is True
    assert trace[0]['object_score_logit']==-2 and not trace[0]['native_object_present']
    assert trace[0]['output']['positive_pixels']==int((out>0).sum())
    assert trace[1]['native']['sha256']==trace[1]['output']['sha256']
    assert '_track_step' not in vars(model)


def test_absent_score_and_suppression_observed_not_gated():
    native=torch.full((1,1,2,2),-1024.); refined=native.clone(); refined[0,0,0,0]=1
    stats=_stats(5,False,torch.tensor([[-3.]]),native,refined,refined,refined)
    assert stats['native_exact_suppressed'] and not stats['native_object_present']
    assert stats['native']['positive_pixels']==0 and stats['output']['positive_pixels']==1
    assert stats['refined']['max']==1 and stats['native']['min']==-1024
    with pytest.raises(ValueError): _stats(0,False,torch.tensor([[float('nan')]]),native,refined,refined,refined)


def test_short_gate_missing_or_incomplete_fails(tmp_path):
    p=tmp_path/'gate.json'; p.write_text(json.dumps({'passed':False}))
    with pytest.raises(ValueError,match='incomplete'): gate.require_gate(p)
    with pytest.raises(ValueError,match='Two full short-run'): gate.verify_mapping({'seq4':{}})


def test_full_video_requires_gate_before_any_inputs(monkeypatch,tmp_path):
    argv=['run','--study',str(tmp_path),'--audit',str(tmp_path),'--split-preflight',str(tmp_path),
          '--checkpoint',str(tmp_path),'--detector',str(tmp_path),'--baseline-reference',str(tmp_path),
          '--seq4-capture',str(tmp_path),'--output',str(tmp_path/'out'),'--sequence','seq4','--arm','A']
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='requires the verified short'): runner.main()
    assert not (tmp_path/'out').exists()


def test_live_environment_and_fork_identity_fail_closed(monkeypatch,tmp_path):
    from modeling import lean_boundary_window_runtime_v1 as runtime
    freeze=tmp_path/'source_freeze.json'; freeze.write_text(json.dumps({'passed': True,'identity':{'fork_revision':'good'}}))
    manifest={'preflight_path':str(tmp_path/'manifest.json'),'reference_capture_path':str(tmp_path),
              'checkpoint_path':str(tmp_path/'checkpoint.pt'),'source_freeze_path':str(freeze)}
    monkeypatch.setattr(runtime,'environment_identity',lambda: {'torch':'actual'})
    monkeypatch.setattr(runtime,'current_identity',lambda *args: {'fork_revision':'good'})
    assert runtime.verify_live_identity(manifest,{'environment':{'torch':'actual'}})[1]['fork_revision']=='good'
    with pytest.raises(ValueError,match='environment'):
        runtime.verify_live_identity(manifest,{'environment':{'torch':'different'}})
    monkeypatch.setattr(runtime,'current_identity',lambda *args: {'fork_revision':'tampered'})
    with pytest.raises(ValueError,match='fork/source'):
        runtime.verify_live_identity(manifest,{'environment':{'torch':'actual'}})


def test_short_factorial_gate_rejects_changed_conditioning(monkeypatch,tmp_path):
    mapping={s:{a:str(tmp_path/s/a) for a in gate.ARMS} for s in gate.SEQS}
    entries={}
    common={'prompt_frame':0,'prompt_sha256':'p','prompt_mask_sha256':'pm','checkpoint_sha256':'c',
            'detector_sha256':'d','candidate_checkpoint_sha256':'x','control_checkpoint_sha256':'y',
            'view':{'kind':'test'},'input_images':[{'frame':0,'sha256':'i'}]}
    initial={'frame':0,'mask_sha256':'mask','pred_masks_sha256':'pred0','maskmem_features_sha256':'mem0',
             'obj_ptr_sha256':'ptr0','object_score_logits_sha256':'score0','output_logits_sha256':'out0'}
    for seq in gate.SEQS:
        for arm in gate.ARMS:
            output='out1' if arm in ('B','D') else 'out0'
            mem='mem1' if arm in ('C','D') else 'mem0'
            first={'frame':1,'mask_sha256':'mask','pred_masks_sha256':'pred0','maskmem_features_sha256':mem,
                   'obj_ptr_sha256':'ptr0','object_score_logits_sha256':'score0','output_logits_sha256':output}
            rec={**common,'masks':[{'frame':0,'sha256':'a'}], 'post_prompt_state':[dict(initial),first]}
            entries[str(Path(mapping[seq][arm]).resolve())]={'folder':mapping[seq][arm],'rec':rec,'runtime':{'passed':True}}
    monkeypatch.setattr(gate,'verify_short_run',lambda folder,seq,arm:entries[str(Path(folder).resolve())])
    monkeypatch.setattr(gate,'read',lambda path: [])
    gate.verify_mapping(mapping)
    entries[str(Path(mapping['seq23']['C']).resolve())]['rec']['post_prompt_state'][0]['obj_ptr_sha256']='tampered'
    with pytest.raises(ValueError,match='conditioning state'):
        gate.verify_mapping(mapping)
