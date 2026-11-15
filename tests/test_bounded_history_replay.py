from copy import deepcopy
import pytest
from scripts.run_wider_history_tome_gate import assert_replay_row


def test_replay_requires_memory_hash_even_when_mask_and_scores_match():
    archived={'frame':32,'prediction_sha256':'mask','stored_sha256':'bank',
              'present':True,'scores':{'dice':0.7},'absent_fp':None,'object_score':8.0}
    assert_replay_row(deepcopy(archived),archived,'seq6')
    changed=deepcopy(archived);changed['stored_sha256']='different'
    with pytest.raises(RuntimeError,match='Archived treatment replay differs'):
        assert_replay_row(changed,archived,'seq6')
    with pytest.raises(RuntimeError,match='Archived treatment replay differs'):
        assert_replay_row(archived,None,'seq6')
