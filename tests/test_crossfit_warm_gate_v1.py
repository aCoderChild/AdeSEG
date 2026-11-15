"""No-GPU checks for warm compatibility gate inputs and viewport handling."""
from scripts import run_crossfit_warm_gate_v1 as gate
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v2 import load_prepared


def test_real_gate_input_geometry_without_annotation_opens():
    guard=install_mask_open_guard(gate.DATA)
    warm=gate.warm_plan();index=gate.validate_plan();existing,_=load_prepared(gate.EXISTING)
    values=gate.gate_inputs('seq3',index,existing,warm)
    assert values[2]==0 and values[3]==list(range(1,15)) and values[7]==0
    values=gate.gate_inputs('seq6',index,existing,warm)
    assert 24 in values[3] and 90 in values[3] and values[7]==0
    values=gate.gate_inputs('seq23',index,existing,warm)
    assert values[4].shape==(720,834) and values[8]==[446,0,1280,720]
    assert values[5][0]>=0 and values[5][2]<=834
    assert guard['open_attempts']==0


def test_gate_study_schedule_uses_same_device_and_includes_late_reads(tmp_path):
    from scripts.run_crossfit_warm_gate_study_v1 import phase_tasks
    tasks=phase_tasks(tmp_path)
    assert [x[0] for x in tasks]==['seq3_capture','seq6_capture','seq23_reference','seq23_capture','audit']
    assert [x[3] for x in tasks]==[14,12,0,1,0]


def test_audit_rejects_changed_historical_tensor(tmp_path):
    import torch
    import pytest
    from scripts.audit_crossfit_warm_gate_v1 import compare_tensors
    from scripts.memory_token_io import digest
    rows=[]
    for name,val in (('new',1),('old',1)):
        directory=tmp_path/name;directory.mkdir();row={'frame':14}
        for key in ('feature','snapshot'):
            path=directory/(key+'.pt');torch.save({'value':torch.tensor([val])},path)
            row[key+'_file']=str(path);row[key+'_sha256']=digest(path)
        rows.append({'observations':[row]})
    compare_tensors(*rows,[14])
    path=tmp_path/'old/snapshot.pt';torch.save({'value':torch.tensor([2])},path)
    with pytest.raises(ValueError,match='tensor changed'):
        compare_tensors(*rows,[14])
    rows[1]['observations'][0]['snapshot_sha256']=digest(path)
    with pytest.raises(ValueError,match='historical tensor differs'):
        compare_tensors(*rows,[14])


def test_old_mps_adapter_identity_and_launcher_sources_are_bound():
    import pytest
    from scripts.memory_token_io import digest
    assert 'scripts/audit_crossfit_warm_gate_v1.py' in gate.SOURCES
    assert 'scripts/run_crossfit_warm_gate_study_v1.py' in gate.SOURCES
    assert 'modeling/lean_indexed_spatial_rope_mps.py' in gate.SOURCES
    assert 'modeling/lean_mps_blocking_pointer_position.py' in gate.SOURCES
    identity={key:digest(gate.ROOT/path) for key,path in (
        ('mps_indexed_sha256','modeling/lean_indexed_spatial_rope_mps.py'),
        ('mps_blocking_sha256','modeling/lean_mps_blocking_pointer_position.py'))}
    gate.verify_old_mps_hooks(identity)
    identity['mps_indexed_sha256']='changed'
    with pytest.raises(ValueError,match='Historical MPS adapter differs'):gate.verify_old_mps_hooks(identity)
