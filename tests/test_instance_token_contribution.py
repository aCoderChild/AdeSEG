"""Instance-routing and cohort-accounting tests; real-frame parity is separate."""
from types import SimpleNamespace

import pytest
import torch

from modeling.instance_token_contribution import InstanceTokenContributionRead, instance_context
from scripts.run_davis_token_contribution import (
    ARMS, PRIMARY, SPLITS, report, score_object, summarize, validate_coverage, video_supplement,
)


def state_fixture():
    return {'obj_id_to_idx':{7:0,42:1},'obj_idx_to_id':{0:7,1:42},
            'output_dict_per_obj':{0:{'native':7},1:{'native':42}}}


def test_replay_routes_the_actual_private_bank_to_persistent_id():
    state=state_fixture(); private={'cloned':'again'}
    proxy=instance_context(state,private,42)
    assert proxy['output_dict_per_obj'][1] is private
    assert proxy['obj_idx_to_id'][1]==42
    assert instance_context(state,state['output_dict_per_obj'][1]) is state
    with pytest.raises(RuntimeError,match='unique'):
        instance_context(state,private)
    with pytest.raises(ValueError,match='prompted'):
        instance_context(state,private,1)


def test_replay_context_cleans_up_after_error_and_rejects_nested_owner():
    controller=InstanceTokenContributionRead.__new__(InstanceTokenContributionRead)
    controller.state=state_fixture();controller.replay_owner=None
    with pytest.raises(RuntimeError,match='fixture'):
        with controller.replaying(42):
            assert controller.replay_owner==42
            with pytest.raises(ValueError,match='nested'):
                with controller.replaying(7):pass
            raise RuntimeError('fixture')
    assert controller.replay_owner is None


def test_instance_cache_signature_excludes_other_objects():
    controller=InstanceTokenContributionRead.__new__(InstanceTokenContributionRead)
    controller.masks={(7,0,2):torch.zeros(8),(42,0,2):torch.ones(8)}
    controller.anchors={(7,0):torch.zeros(2),(42,0):torch.ones(2)}
    controller.usage={}
    own=controller.cache_signature(7);whole=controller.cache_signature()
    controller.masks[(42,0,2)]+=1
    assert controller.cache_signature(7)==own
    assert controller.cache_signature()!=whole


@pytest.mark.parametrize('owner,removed',[(7,256),(42,0)])
def test_asymmetric_intervention_preserves_non_target_owner(owner,removed):
    controller=InstanceTokenContributionRead.__new__(InstanceTokenContributionRead)
    controller.state=state_fixture();controller.only_owner=7
    controller.context={'frame':4,'object_id':owner,
        'spans':[{'frame':0,'t_pos':0,'start':0,'stop':4096},
                 {'frame':2,'t_pos':1,'start':4096,'stop':8192},
                 {'frame':3,'t_pos':2,'start':8192,'stop':12288}],
        'probabilities':torch.full((1,3,64,64),.3),
        'outputs':[{'maskmem_features':torch.ones(1,64,64,64)} for _ in range(3)]}
    controller.selector_mode='regional';controller.keep_fraction=.25
    controller.arm='rejected';controller.sequence='fixture'
    controller.masks={(owner,0,2):torch.arange(4096)<1024}
    controller.usage={(owner,0,2):torch.full((4096,),1/4096)}
    controller.selection_layers=[]
    module=SimpleNamespace(training=False)
    controller._before_selection(0,module,(),{'v':torch.zeros(1,12292,64),'num_k_exclude_rope':4})
    assert controller.arm=='rejected'
    assert controller.selection_layers[-1]['object_id']==owner
    assert controller.selection_layers[-1]['counts']==[4096,4096-removed,4096]
    assert controller.selection_layers[-1]['pointer_keys']==4


def test_dice_uses_instance_ids_and_ignores_void_separately_from_official_jf():
    import numpy as np
    labels=np.array([[7,0],[255,42]],dtype=np.uint8)
    prediction=np.array([[True,False],[True,False]])
    assert score_object(prediction,labels,7)['dice']==1
    assert score_object(prediction,labels,42)['dice']==0
    absent=score_object(prediction,labels,9)
    assert not absent['present'] and absent['false_alarm']


def cohort_rows():
    plan={'7':{'present':[3],'absent':[]},'42':{'present':[3],'absent':[]}}
    rows={a:[] for a in ARMS}
    for arm in ARMS:
        for frame in ([3] if arm in SPLITS else range(5)):
            for oid in (7,42):
                rows[arm].append({'frame':frame,'object_id':oid,'object_ids':[7,42],
                    'official_scored':0<frame<4,'sampled':frame==3,
                    'present':True,'false_alarm':None,'dice':.5 if oid==7 else 1.})
    return rows,plan


def test_coverage_requires_every_object_and_full_indexed_timeline():
    rows,plan=cohort_rows()
    validate_coverage(rows,[7,42],plan,5)
    result=video_supplement(rows['native'],[7,42],plan)
    assert result['sampled_present_dice']==.75
    assert result['per_object']['7']['official_frames']==3
    rows['random'].pop()
    with pytest.raises(RuntimeError,match='coverage'):
        validate_coverage(rows,[7,42],plan,5)


def test_coverage_rejects_duplicate_split_or_changed_denominators():
    rows,plan=cohort_rows();rows['rejected_a'].append(rows['rejected_a'][0])
    with pytest.raises(RuntimeError,match='union'):
        validate_coverage(rows,[7,42],plan,5)
    rows,plan=cohort_rows();rows['retained'][3]['present']=False
    with pytest.raises(RuntimeError,match='denominators'):
        validate_coverage(rows,[7,42],plan,5)


def test_statistics_use_video_units_not_number_of_instances():
    results={};official={a:{'global':{},'per_video':{}} for a in PRIMARY}
    for name,value in [('two_objects',.5),('one_object',1.)]:
        results[name]={a:{'sampled_present_dice':value,'full_present_dice':value} for a in ARMS}
        for arm in PRIMARY:
            official[arm]['per_video'][name]={'J&F-Mean':value}
    result=summarize(results,official)
    assert result['means']['native']['equal_video_jf']==.75
    assert len(result['holm_pvalues_four_contrasts'])==4
    assert all(c['n_videos']==2 for c in result['primary_contrasts'].values())
    assert not result['ranking_discrimination_signal']  # partial cohorts cannot pass


def test_report_distinguishes_official_global_and_video_unit_metrics(tmp_path):
    import json
    results={};official={a:{'global':{'J-Mean':.4,'F-Mean':.6,'J&F-Mean':.5},'per_video':{}} for a in PRIMARY}
    for name,value in [('video_a',.5),('video_b',1.)]:
        results[name]={a:{'sampled_present_dice':value,'full_present_dice':value} for a in ARMS}
        for arm in PRIMARY:official[arm]['per_video'][name]={'J&F-Mean':value}
    old=tmp_path/'prior.json'
    old.write_text(json.dumps({'primary_contrasts':{
        f'rejected_minus_{a}':{'mean':-.01} for a in ('retained','random')}}))
    report(tmp_path,summarize(results,official),{'protocol':{'polypgen_reference':{'results':str(old)}}})
    text=(tmp_path/'analysis.md').read_text()
    assert '0.500000 | 0.750000' in text
    assert 'not passed' in text and '| video_b |' in text
