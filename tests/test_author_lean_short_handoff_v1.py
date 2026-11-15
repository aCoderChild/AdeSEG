import json
import pytest
from scripts import verify_author_lean_short_handoff_v1 as mod


def setup_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'OUT', tmp_path)
    folder=tmp_path/'batch'; folder.mkdir()
    for name in ['short_only_launcher.py','phases.jsonl','gate.json']:
        (folder/name).write_text(name)
    sha=mod.digest(folder/'short_only_launcher.py')
    (folder/'run_manifest.json').write_text(json.dumps({'short_only_launcher_sha256':sha}))
    data={'launcher_snapshot':str(folder/'short_only_launcher.py'),'launcher_sha256':sha,
          'manifest_sha256':mod.digest(folder/'run_manifest.json'),'phases_sha256':mod.digest(folder/'phases.jsonl'),
          'gate_sha256':mod.digest(folder/'gate.json'),'full_resume_requires_independent_handoff_validation':True}
    (folder/'short_handoff.json').write_text(json.dumps(data))
    monkeypatch.setattr(mod,'artifact',lambda folder,name: folder/'gate.json')
    monkeypatch.setattr(mod,'validate_resume',lambda f:(f, [(x,[]) for x in ('smoke_seq6','smoke_seq23','gate','full_seq2')], data['manifest_sha256'],3))
    return folder


def test_handoff_accepts_exact_short_evidence(tmp_path,monkeypatch):
    assert mod.verify(setup_proof(tmp_path,monkeypatch))['passed']


@pytest.mark.parametrize('name',['short_only_launcher.py','phases.jsonl','gate.json','run_manifest.json'])
def test_handoff_rejects_mutated_artifacts(tmp_path,monkeypatch,name):
    root=setup_proof(tmp_path,monkeypatch)
    (root/name).write_text('tampered')
    with pytest.raises(ValueError): mod.verify(root)


def test_handoff_rejects_incomplete_or_extended_prefix(tmp_path,monkeypatch):
    root=setup_proof(tmp_path,monkeypatch)
    monkeypatch.setattr(mod,'validate_resume',lambda f:(f,[],mod.digest(f/'run_manifest.json'),2))
    with pytest.raises(ValueError): mod.verify(root)
