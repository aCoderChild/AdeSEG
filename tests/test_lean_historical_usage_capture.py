from types import SimpleNamespace
import pytest
import torch
from modeling.lean_historical_usage_capture import capture_historical_usage

def test_usage_observer_preserves_author_call_and_records_indices(tmp_path):
    calls=[];model=SimpleNamespace(_dump_retained_tokens_record=lambda **kw:calls.append(kw));events=[]
    usage=torch.arange(4096,dtype=torch.float32);ids=torch.arange(1024)
    with capture_historical_usage(model,tmp_path,events):
        model._dump_retained_tokens_record(frame_idx=8,latest_prev={'frame_idx':7},usage_scores=usage,keep_idx=ids)
    x=torch.load(tmp_path/'historical_usage/000007.pt',weights_only=True)
    assert x['read_frame']==8 and x['source_frame']==7 and torch.equal(x['usage'],usage) and torch.equal(x['keep_idx'],ids)
    assert len(calls)==1 and events[0]['source_frame']==7

def test_usage_observer_rejects_duplicate_or_invalid_source(tmp_path):
    model=SimpleNamespace(_dump_retained_tokens_record=lambda **kw:None);events=[];usage=torch.zeros(4096);ids=torch.arange(1024)
    with capture_historical_usage(model,tmp_path,events):
        model._dump_retained_tokens_record(frame_idx=8,latest_prev={'frame_idx':7},usage_scores=usage,keep_idx=ids)
        with pytest.raises(ValueError):model._dump_retained_tokens_record(frame_idx=9,latest_prev={'frame_idx':7},usage_scores=usage,keep_idx=ids)
