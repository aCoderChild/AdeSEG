import copy

import pytest
import torch
import torch.nn.functional as F

from scripts.preflight_mixed_refiner_v1 import TARGETS, canonical_sha, require_no_cross_split_image_overlap, validate_samples
from scripts.train_region_mixed_loss_v1 import ARMS, LOSSES, model_for, pixel_loss, state_sha256, validate_parity_checks
from scripts.train_multilevel_refiner_seq4_empty_v2 import loss_fn
import scripts.score_region_mixed_loss_v1 as scorer


def samples():
    rows=[]
    for seq in ('seq2','seq3'):
        rows.extend({'sequence':seq,'frame':i,'split':'train','capture_device':'cpu'} for i in range(5))
    rows.extend({'sequence':'seq4','frame':i,'split':'train','capture_device':'cpu'} for i in (10,17,18,19,21,24,28,38,47))
    for seq in ('seq6','seq13'):
        rows.extend({'sequence':seq,'frame':i,'split':'train','capture_device':'mps'} for i in TARGETS[seq])
    for seq in ('seq9','seq10'):
        rows.extend({'sequence':seq,'frame':i,'split':'validation','capture_device':'cpu'} for i in range(5))
    return rows


def test_mixed_manifest_keeps_devices_and_held_out_sequences_separate():
    rows=samples();validate_samples(rows)
    bad=copy.deepcopy(rows);bad[0]['capture_device']='mps'
    with pytest.raises(ValueError,match='capture device'):validate_samples(bad)
    bad=copy.deepcopy(rows);bad[-1]=copy.deepcopy(bad[0])
    with pytest.raises(ValueError,match='unique'):validate_samples(bad)
    require_no_cross_split_image_overlap({'seq6':{'a'}},{'seq9':{'b'}})
    with pytest.raises(ValueError,match='overlap'):
        require_no_cross_split_image_overlap({'seq6':{'same'}},{'seq9':{'same'}})


def test_composite_identity_is_canonical_and_device_specific():
    identity={'schema':'mixed','cpu':'a','mps':'b'}
    assert canonical_sha(identity)==canonical_sha({'mps':'b','cpu':'a','schema':'mixed'})
    assert canonical_sha(identity)!=canonical_sha({'schema':'mixed','cpu':'a','mps':'other'})


def test_parity_receipt_requires_all_cpu_and_mps_capture_cases_and_finite_gradients():
    checks=[{'sequence':seq,'frame':frame,'loss_kind':kind,'capture_device':device,
             'cpu_loss':1.0,'mps_loss':1.0,'maximum_gradient_difference':0.0,'maximum_gradient_magnitude':0.1}
            for seq,frame,device in (('seq4',24,'cpu'),('seq4',21,'cpu'),('seq6',90,'mps'),('seq13',62,'mps')) for kind in LOSSES]
    validate_parity_checks(checks)
    with pytest.raises(ValueError,match='coverage'):validate_parity_checks(checks[:-1])
    changed=copy.deepcopy(checks);changed[0]['maximum_gradient_difference']=float('nan')
    with pytest.raises(ValueError,match='numerical'):validate_parity_checks(changed)


def test_ordinary_bce_matches_original_and_robust_limits_suppressed_gradient():
    prediction=torch.tensor([[[[-1024.,0.,3.]]]],requires_grad=True)
    target=torch.ones_like(prediction)
    ordinary=pixel_loss(prediction,target,'bce')
    torch.testing.assert_close(ordinary,loss_fn(prediction,target))
    ordinary.backward(retain_graph=True);grad=prediction.grad.detach().clone();prediction.grad.zero_()
    robust=pixel_loss(prediction,target,'robust_bce');robust.backward()
    assert torch.isfinite(robust) and robust<ordinary
    assert prediction.grad[0,0,0,0].abs()<grad[0,0,0,0].abs()/50
    b=F.binary_cross_entropy_with_logits(prediction.detach(),target,reduction='none')
    probability=prediction.detach().sigmoid()
    dice=1-(2*(probability*target).sum()+1)/(probability.sum()+target.sum()+1)
    torch.testing.assert_close(robust.detach(),(10*torch.log1p(b/10)).mean()+dice)


def test_arm_parameters_match_and_unknown_loss_fails():
    assert len({sum(p.numel() for p in model_for(arm).parameters()) for arm in ARMS})==1
    with pytest.raises(ValueError,match='Unknown'):
        pixel_loss(torch.zeros(1),torch.zeros(1),'other')


def test_all_arms_can_share_exact_initialized_weights():
    fingerprints=[]
    for arm in ARMS:
        torch.manual_seed(20261114)
        fingerprints.append(state_sha256(model_for(arm)))
    assert len(set(fingerprints))==1


def test_scorer_does_not_open_labels_if_any_arm_is_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(scorer,'mixed_manifest',lambda _:({'samples':[]},'manifest'))
    monkeypatch.setattr(scorer,'environment_identity',lambda:{})
    monkeypatch.setattr(scorer,'verify_plan',lambda *args:None)
    calls=[];labels=[]
    def check(*args):
        calls.append((args[1],args[2]))
        if len(calls)==8:raise ValueError('eighth arm missing')
        return {}
    monkeypatch.setattr(scorer,'check_run',check)
    monkeypatch.setattr(scorer,'load_target',lambda *args:labels.append(args))
    with pytest.raises(ValueError,match='eighth arm'):
        scorer.evaluate(tmp_path,tmp_path/'plan',tmp_path/'parity',tmp_path/'runs',tmp_path/'out')
    assert len(calls)==8 and not labels


def test_scorer_requires_each_epoch_to_match_manifest_training_keys():
    original=[x for x in samples() if x['split']=='train']
    history=[{'epoch':i,'samples':[{**x,'loss':1.0,'maximum_gradient':2.0} for x in original]}
             for i in range(1,scorer.EPOCHS+1)]
    assert len(scorer.validate_training_order(history,{'samples':samples()}))==scorer.EPOCHS
    wrong=copy.deepcopy(history);wrong[0]['samples'][0]['frame']=99999
    with pytest.raises(ValueError,match='coverage'):
        scorer.validate_training_order(wrong,{'samples':samples()})
