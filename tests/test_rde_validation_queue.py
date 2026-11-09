import json
import pytest
from scripts.memory_token_io import digest
from scripts.queue_rde_value_validation import check_training_process,ready_training,check_cross_pipeline


def test_missing_or_reused_pid_cannot_launch_validation(tmp_path):
    assert not check_training_process(None,tmp_path)
    assert check_training_process(f'python scripts/run_rde_value_training.py --output-dir {tmp_path}',tmp_path)
    for command in ['python unrelated.py',f'python scripts/run_rde_value_training.py --output-dir {tmp_path}/other']:
        with pytest.raises(RuntimeError):check_training_process(command,tmp_path)


def test_missing_terminal_receipt_or_changed_identity_cannot_release(tmp_path):
    path=tmp_path/'identity.json';path.write_text('{}');sha=digest(path)
    with pytest.raises(FileNotFoundError):ready_training(tmp_path,sha)
    path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='identity'):ready_training(tmp_path,sha)


def test_only_complete_arms_with_matching_final_files_can_release(tmp_path):
    def complete(directory,result):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'results.json').write_text(json.dumps(result))
        (directory/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(directory/'results.json')}))
    (tmp_path/'identity.json').write_text('{}');sha=digest(tmp_path/'identity.json');arms={}
    for arm in ['segmentation_only','segmentation_plus_consistency']:
        directory=tmp_path/'training'/arm;directory.mkdir(parents=True)
        (directory/'adapter_final.pt').write_bytes(b'fixture weights')
        arms[arm]={'final_adapter_file_sha256':digest(directory/'adapter_final.pt')}
        complete(directory,arms[arm])
    complete(tmp_path,{'engineering':False,'arms':arms});ready_training(tmp_path,sha)
    final=tmp_path/'training/segmentation_plus_consistency/adapter_final.pt';final.write_bytes(b'changed weights')
    with pytest.raises(RuntimeError,match='checkpoint'):ready_training(tmp_path,sha)
    complete(tmp_path,{'engineering':True,'arms':arms})
    with pytest.raises(RuntimeError,match='Engineering'):ready_training(tmp_path,sha)


def test_cross_pipeline_proof_must_match_sources_and_native_reference(tmp_path,monkeypatch):
    import scripts.queue_rde_value_validation as queue
    monkeypatch.setattr(queue,'gate_sources',lambda:{'current':'source'})
    native=tmp_path/'native';native.mkdir();(native/'results.json').write_text('{}')
    gate=tmp_path/'gate';gate.mkdir()
    def write(result):
        (gate/'results.json').write_text(json.dumps(result))
        (gate/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(gate/'results.json')}))
    result={'cross_pipeline_parity_passed':True,'sources':{'current':'source'},'matched_native_results_sha256':digest(native/'results.json')}
    write(result);assert check_cross_pipeline(gate,native)==digest(gate/'results.json')
    write({**result,'sources':{}})
    with pytest.raises(RuntimeError,match='stale'):check_cross_pipeline(gate,native)
    write({**result,'matched_native_results_sha256':'wrong'})
    with pytest.raises(RuntimeError,match='another native'):check_cross_pipeline(gate,native)
