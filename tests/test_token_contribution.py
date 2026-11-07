from types import SimpleNamespace

import pytest
import torch

from modeling.token_contribution import ARMS, TokenContributionRead, block_plan
from scripts.run_token_contribution import spaced, summarize, native_signatures


def test_groups_preserve_shape_count_and_split_union():
    retained = torch.zeros(4096, dtype=torch.bool)
    retained[:1024] = True
    plans, info = block_plan(retained, 'fixture')
    assert torch.equal(plans['rejected_a'] | plans['rejected_b'], plans['rejected'])
    assert not (plans['rejected_a'] & plans['rejected_b']).any()
    assert not (plans['rejected'] & plans['retained']).any()
    for arm in plans:
        assert plans[arm].sum() == (128 if arm.endswith(('_a', '_b')) else 256)
        cells = plans[arm].reshape(64,64).unfold(0,4,4).unfold(1,4,4).reshape(-1,16)
        assert ((cells.sum(1) == 0) | (cells.sum(1) == 16)).all()
    assert info['rejected']['retained_fraction'] == 0
    assert info['retained']['retained_fraction'] == 1


def test_repeatability_and_context_dependent_random_control():
    retained = torch.arange(4096) % 4 == 0
    a, info = block_plan(retained,'same')
    b, repeated = block_plan(retained,'same')
    c, _ = block_plan(retained,'different')
    assert info == repeated
    assert all(torch.equal(a[k],b[k]) for k in a)
    assert not torch.equal(a['random'],c['random'])
    assert info['retained']['retained_fraction'] == .25  # honest tie, not false enrichment


@pytest.mark.parametrize('block,groups',[(3,16),(4,15),(4,0),(4,130)])
def test_invalid_group_geometry_rejected(block,groups):
    with pytest.raises(ValueError):
        block_plan(torch.zeros(4096,dtype=torch.bool),'x',block=block,groups=groups)


def fixture_rows():
    d = {'native':.8,'rejected_a':.79,'rejected_b':.79,'rejected':.75,'retained':.7,'random':.76}
    return {'seq2':{a:[{'frame':3,'present':True,'scores':{'dice':v,'iou':v,'boundary_f':v},'absent_fp':None},
                       {'frame':4,'present':False,'scores':{'dice':1.,'iou':1.,'boundary_f':1.},'absent_fp':False}]
                    for a,v in d.items()}}


def test_utility_interaction_sign_and_primary_denominators():
    result = summarize(fixture_rows())
    assert result['interaction_mean'] == pytest.approx(.03)
    assert result['means']['native']['dice'] == .8
    assert result['means']['native']['empty_frames'] == 1
    assert result['primary_contrasts']['rejected_minus_retained']['mean'] == pytest.approx(.05)
    assert not result['selector_discrimination_signal']  # partial cohort is never a pass


def test_summary_rejects_unmatched_targets_or_presence():
    rows = fixture_rows()
    rows['seq2']['random'][0]['frame'] = 5
    with pytest.raises(RuntimeError,match='targets'):
        summarize(rows)
    rows = fixture_rows()
    rows['seq2']['random'][0]['present'] = False
    with pytest.raises(RuntimeError,match='denominators'):
        summarize(rows)


def test_sampling_is_even_and_without_duplicates():
    assert spaced(list(range(20)),4) == [0,6,13,19]
    assert spaced([3,5],4) == [3,5]
    assert spaced([],4) == []


@pytest.mark.parametrize('arm',('collect',*ARMS))
def test_hook_preserves_prompt_latest_and_matches_requested_group_budget(arm):
    controller = TokenContributionRead.__new__(TokenContributionRead)
    controller.context = {'frame':4,'object_id':None,
        'spans':[{'frame':0,'t_pos':0,'start':0,'stop':4096},
                 {'frame':2,'t_pos':1,'start':4096,'stop':8192},
                 {'frame':3,'t_pos':2,'start':8192,'stop':12288}],
        'probabilities':torch.full((1,3,64,64),.3),
        'outputs':[{'maskmem_features':torch.ones(1,64,64,64)} for _ in range(3)]}
    controller.selector_mode = 'regional'
    controller.keep_fraction = .25
    controller.arm, controller.sequence = arm,'seq2'
    controller.masks = {(None,0,2):torch.arange(4096)<1024}
    controller.usage = {(None,0,2):torch.full((4096,),1/4096)}
    controller.selection_layers = []
    module = SimpleNamespace(training=False)
    controller._before_selection(0,module,(),{'v':torch.zeros(1,12292,64),'num_k_exclude_rope':4})
    removed = 0 if arm in ('collect','native') else (128 if arm.endswith(('_a','_b')) else 256)
    assert controller.selection_layers[-1]['counts'] == [4096,4096-removed,4096]
    assert controller.selection_layers[-1]['pointer_keys'] == 4
    assert module._sparse_mask.numel() == 4096
    assert module._sparse_mask.sum() == 4096-removed


def test_target_replay_does_not_reestimate_selector_or_change_cache():
    controller = TokenContributionRead.__new__(TokenContributionRead)
    controller.arm,controller.current_pointers = 'random',2
    controller.selection_layers = [{'spatial_keys_after':6}]
    q,k,v = torch.randn(1,1,3,4),torch.randn(1,1,8,4),torch.randn(1,1,8,4)
    actual = controller._capture_sdpa(q,k,v)
    assert torch.equal(actual,torch.nn.functional.scaled_dot_product_attention(q,k,v))
    with pytest.raises(RuntimeError,match='key count'):
        controller._capture_sdpa(q,k[:,:,:7],v[:,:,:7])


def test_independent_reference_includes_pointer_score_and_native_low_resolution():
    stored = {k:torch.ones(1,1) for k in ('pred_masks','obj_ptr','object_score_logits')}
    state = {'output_dict_per_obj':{0:{'cond_frame_outputs':{0:stored},'non_cond_frame_outputs':{1:stored}}}}
    full = torch.ones(1,1)
    original = native_signatures(state,1,0,full)
    for key in stored:
        stored[key] += 1
        assert native_signatures(state,1,0,full) != original
        stored[key] -= 1
    assert native_signatures(state,0,0,full) == original
