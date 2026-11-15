import json

import pytest
import torch

from scripts.queue_fixed_assignment_attention_gate import verify_cohort_completion,SEQUENCES,ARMS
from scripts.run_fixed_assignment_attention_gate import require_same_prediction


def completed_cohort(folder):
    (folder/"completion.json").write_text(json.dumps({"complete":True,
        "sequences":sorted(SEQUENCES),"arms":sorted(ARMS)}))
    (folder/"progress.json").write_text(json.dumps({"status":"complete"}))
    (folder/"results.json").write_text(json.dumps({"sequence_summaries":{
        arm:{seq:{} for seq in SEQUENCES} for arm in ARMS|{"native"}}}))


def test_queue_requires_terminal_cohort_and_exact_21_sequence_ledger(tmp_path):
    completed_cohort(tmp_path)
    assert set(verify_cohort_completion(tmp_path))=={"completion.json","results.json"}
    (tmp_path/"progress.json").write_text(json.dumps({"status":"running"}))
    with pytest.raises(RuntimeError,match="not complete"):
        verify_cohort_completion(tmp_path)


def test_queue_rejects_missing_sequence_even_with_complete_marker(tmp_path):
    completed_cohort(tmp_path)
    p=tmp_path/"results.json";data=json.loads(p.read_text())
    del data["sequence_summaries"]["uniform_generic"]["seq23"]
    p.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match="ledger is incomplete"):
        verify_cohort_completion(tmp_path)


@pytest.mark.parametrize("changed",["pred_masks","object_score_logits"])
def test_prediction_parity_checks_mask_and_presence(changed):
    native={"pred_masks":torch.ones(1,1,2,2),"object_score_logits":torch.ones(1,1)}
    actual={name:value.clone() for name,value in native.items()}
    require_same_prediction(native,actual,"fixture")
    actual[changed].zero_()
    with pytest.raises(RuntimeError,match=changed):
        require_same_prediction(native,actual,"fixture")
