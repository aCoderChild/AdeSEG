from copy import deepcopy
import pytest
from scripts.run_history_compression_counterfactual import validate_matched_reads


def reads():
    return [{'layer':i,'pointer_keys':4,'spatial_keys_before':32768,
             'spatial_keys_after':28672,'spatial_key_mass_sum':32768.0,
             'frames':[{'frame':3,'t_pos':0,'source_tokens':4096},
                       {'frame':7,'t_pos':6,'source_tokens':4096}]}
            for i in range(4)]


def test_counterfactual_rejects_different_source_frame_with_same_budget():
    dense=reads();compressed=deepcopy(dense)
    compressed[2]['frames'][1]['frame']=6
    with pytest.raises(RuntimeError,match='source frames differ'):
        validate_matched_reads(dense,compressed)


def test_counterfactual_accepts_matching_sources_but_rejects_pointer_mismatch():
    dense=reads();compressed=deepcopy(dense)
    validate_matched_reads(dense,compressed)
    compressed[0]['pointer_keys']=8
    with pytest.raises(RuntimeError,match='pointer counts differ'):
        validate_matched_reads(dense,compressed)


def test_counterfactual_rejects_missing_layer_and_mass_loss():
    with pytest.raises(RuntimeError,match='four layers'):
        validate_matched_reads(reads()[:-1],reads())
    compressed=reads();compressed[0]['spatial_key_mass_sum']=1
    with pytest.raises(RuntimeError,match='loses spatial mass'):
        validate_matched_reads(reads(),compressed)


def test_dense_passthrough_receipt_has_no_per_frame_token_count():
    dense=reads();compressed=deepcopy(dense)
    for layer in dense:
        for frame in layer['frames']:
            frame.pop('source_tokens')
    validate_matched_reads(dense,compressed)
    compressed[0]['spatial_keys_before']+=4096
    with pytest.raises(RuntimeError,match='input spatial counts differ'):
        validate_matched_reads(dense,compressed)
