from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import run_rde_value_training as runner


def args(tmp_path,**fields):
    result=dict(output_dir=tmp_path/'output',gates_root=tmp_path/'gates',rde_repository=tmp_path/'rde',seed=20261109,
                engineering=True,interrupt_before_checkpoint=0,runner_gate=None,native_reference=None)
    result.update(fields)
    return SimpleNamespace(**result)


def test_engineering_prefix_is_fixed_train_only_and_full_mode_is_not_truncated(monkeypatch):
    videos={'seq2':list(range(20)),'seq3':list(range(30)),'seq13':list(range(40))}
    design={'data':{'train_sequences':['seq2','seq3']}}
    monkeypatch.setattr(runner,'prompt_index',lambda *a:2)
    assert runner.selected_frames(videos,design,True)=={'seq2':[2,3,4,5,6],'seq3':[2,3,4,5,6]}
    assert runner.selected_frames(videos,design,False)=={k:videos[k] for k in ['seq2','seq3']}


def test_missing_safety_receipt_blocks_full_mode_before_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'frozen_sources',lambda:{})
    monkeypatch.setattr(runner,'verify_receipt',lambda *a:({},'hash'))
    a=args(tmp_path,engineering=False)
    with pytest.raises(RuntimeError,match='requires runner gate'):runner.run(a,{})
    assert not a.output_dir.exists()


def test_replication_seed_is_not_opened_without_a_primary_release(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'frozen_sources',lambda:{})
    monkeypatch.setattr(runner,'verify_receipt',lambda *a:({},'hash'))
    a=args(tmp_path,seed=20261110)
    with pytest.raises(RuntimeError,match='Replication seeds'):runner.run(a,{})
    assert not a.output_dir.exists()
