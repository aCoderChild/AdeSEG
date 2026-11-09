import pytest
import torch
from scripts.memory_value_query import spatial_values


def fixture():
    def item(n): return {'maskmem_features': torch.full((1, 2, 2, 2), float(n))}
    bank = {'cond_frame_outputs': {0: item(1)}, 'non_cond_frame_outputs': {2: item(2), 3: item(3)}}
    spans = [{'frame': frame, 'start': i*4, 'stop': (i+1)*4} for i, frame in enumerate([0, 2, 3])]
    teachers = {f: {'mask_only': torch.full((1, 2, 2, 2), float(10*f))} for f in [2, 3]}
    return bank, spans, teachers


def test_teacher_replaces_only_nonconditioning_frames_in_native_order():
    bank, spans, teachers = fixture()
    value = spatial_values(bank, spans, teachers, 'mask_only')
    assert value.shape == (12, 1, 2)
    assert torch.equal(value[:, 0, 0], torch.tensor([1.]*4 + [20.]*4 + [30.]*4))
    assert (bank['non_cond_frame_outputs'][2]['maskmem_features'] == 2).all()
    assert (spatial_values(bank, spans)[:4] == 1).all()


def test_missing_teacher_rejected_instead_of_using_another_source():
    bank, spans, teachers = fixture()
    del teachers[2]
    with pytest.raises(KeyError): spatial_values(bank, spans, teachers, 'mask_only')


def test_misaligned_span_and_teacher_dtype_rejected():
    bank, spans, teachers = fixture()
    spans[1]['start'] = 5
    with pytest.raises(RuntimeError): spatial_values(bank, spans)
    bank, spans, teachers = fixture()
    teachers[2]['mask_only'] = teachers[2]['mask_only'].double()
    with pytest.raises(RuntimeError): spatial_values(bank, spans, teachers, 'mask_only')
