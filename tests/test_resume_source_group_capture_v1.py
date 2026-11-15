from collections import namedtuple
from pathlib import Path
import subprocess,sys
import pytest
import scripts.resume_source_group_capture_v1 as resume


def test_phase_logs_are_never_overwritten(tmp_path):
    log=tmp_path/'phase.log'
    resume.run_phase([sys.executable,'-c','print("first")'],log,{})
    assert log.read_text().strip()=='first'
    with pytest.raises(FileExistsError):resume.run_phase([sys.executable,'-c','print("second")'],log,{})
    assert log.read_text().strip()=='first'


def test_guard_stops_before_creating_output_directory(monkeypatch,tmp_path):
    pre={'sequences':{}};config={'min_free_bytes':22}
    monkeypatch.setattr(resume,'environment_setup',lambda:(pre,config,'sha'))
    monkeypatch.setattr(resume,'discover_completed',lambda *_:{name:[{}] for name in resume.PRIOR})
    monkeypatch.setattr(resume,'discover_references',lambda *_:{})
    usage=namedtuple('usage','total used free')(100,79,21)
    monkeypatch.setattr(resume.shutil,'disk_usage',lambda _:usage)
    monkeypatch.setattr(resume.tempfile,'mkdtemp',lambda **_:pytest.fail('created output before guard'))
    monkeypatch.setattr(sys,'argv',['resume','--execute'])
    with pytest.raises(RuntimeError,match='below original'):resume.main()
    assert list(tmp_path.iterdir())==[]


def test_discovery_skips_verified_completed_only(monkeypatch,tmp_path):
    a=tmp_path/'source_group_full_capture_a';a.mkdir();b=tmp_path/'source_group_capture_resume_v1_b';b.mkdir()
    ignored=tmp_path/'unrelated';ignored.mkdir()
    monkeypatch.setattr(resume,'validate_existing',lambda root,seq,*_: {'ok':str(root)} if root==a and seq=='seq12' else None)
    result=resume.discover_completed(tmp_path,{},'sha')
    assert set(result)=={'seq12'} and len(result['seq12'])==1


def test_monitor_reports_failure_and_readiness(tmp_path):
    import threading
    ready=threading.Event();stop=threading.Event();errors=[]
    (tmp_path/'free_space_monitor.jsonl').write_text('do not overwrite')
    resume.monitor_space(tmp_path,22,stop,ready,errors)
    assert ready.is_set() and errors
    assert (tmp_path/'free_space_monitor.jsonl').read_text()=='do not overwrite'


def test_monitor_records_space_without_waiting_when_stopped(tmp_path):
    import json,threading
    ready=threading.Event();stop=threading.Event();stop.set();errors=[]
    resume.monitor_space(tmp_path,22,stop,ready,errors)
    assert ready.is_set() and not errors
    rows=(tmp_path/'free_space_monitor.jsonl').read_text().splitlines()
    assert len(rows)==1 and json.loads(rows[0])['guard_bytes']==22


def test_gate_binding_rejects_changed_identity_or_gate():
    gate={'identity':{'model':'a'}};audit={'identity':{'model':'a'}}
    ref={'identity':{'model':'a'},'short_gate_sha256':'g'};cap=dict(ref)
    resume.validate_gate_binding(audit,ref,cap,gate,'g')
    cap['short_gate_sha256']='other'
    with pytest.raises(ValueError,match='short gate'):resume.validate_gate_binding(audit,ref,cap,gate,'g')
    cap=dict(ref);audit={'identity':{'model':'changed'}}
    with pytest.raises(ValueError,match='short gate'):resume.validate_gate_binding(audit,ref,cap,gate,'g')


def test_conditioning_must_match_archived_prompt(tmp_path):
    from PIL import Image
    from scripts.memory_token_io import digest
    source=tmp_path/'source';archive=source/'seq12';archive.mkdir(parents=True)
    img=Image.new('L',(3,3),0);img.putpixel((1,1),255);img.save(archive/'prompt_refined.png')
    dirs=[tmp_path/'ref',tmp_path/'cap']
    for d in dirs:d.mkdir();img.save(d/'conditioning_mask.png')
    spec={'source':str(source),'archived_prompt_mask_sha256':digest(archive/'prompt_refined.png')}
    resume.validate_archived_conditioning(spec,'seq12',dirs[0]/'receipt.json',dirs[1]/'receipt.json')
    Image.new('L',(3,3),0).save(dirs[1]/'conditioning_mask.png')
    with pytest.raises(ValueError,match='Conditioning'):resume.validate_archived_conditioning(spec,'seq12',dirs[0]/'receipt.json',dirs[1]/'receipt.json')


def test_interprocess_lock_rejects_concurrent_resume(tmp_path):
    lock=tmp_path/'resume.lock'
    code='from scripts.resume_source_group_capture_v1 import exclusive_lock\nimport sys\nwith exclusive_lock(sys.argv[1]): pass'
    with resume.exclusive_lock(lock):
        other=subprocess.run([sys.executable,'-c',code,str(lock)],capture_output=True,text=True)
    assert other.returncode!=0 and 'already running' in other.stderr
    with resume.exclusive_lock(lock):pass


def test_completed_study_succeeds_below_guard_without_output(monkeypatch):
    pre={'sequences':{}};config={'min_free_bytes':22}
    monkeypatch.setattr(resume,'environment_setup',lambda:(pre,config,'sha'))
    monkeypatch.setattr(resume,'discover_completed',lambda *_:{name:[{}] for name in resume.ALL_SEQUENCES})
    monkeypatch.setattr(resume,'discover_references',lambda *_:{})
    usage=namedtuple('usage','total used free')(100,99,1)
    monkeypatch.setattr(resume.shutil,'disk_usage',lambda _:usage)
    monkeypatch.setattr(resume.tempfile,'mkdtemp',lambda **_:pytest.fail('created output despite completion'))
    monkeypatch.setattr(sys,'argv',['resume','--execute'])
    resume.main()


def test_reusable_reference_revalidates_all_saved_masks(monkeypatch,tmp_path):
    import json
    from PIL import Image
    from scripts.memory_token_io import digest
    monkeypatch.setattr(resume,'OUT',tmp_path)
    prepath=tmp_path/'pre';prepath.mkdir();(prepath/'manifest.json').write_text('{}')
    monkeypatch.setattr(resume,'PRE',prepath)
    src=tmp_path/'archived'/'seq12';src.mkdir(parents=True)
    original=Image.new('L',(2,2),0);original.putpixel((0,0),255);original.save(src/'prompt_refined.png')
    run=tmp_path/'source_group_capture_resume_v1_x';refdir=run/'seq12_reference';refdir.mkdir(parents=True)
    original.save(refdir/'conditioning_mask.png')
    rows=[]
    for i in range(3):
        path=refdir/f'{i:06d}.png';path.write_bytes(str(i).encode());rows.append({'frame':i,'mask_sha256':digest(path)})
    identity={'preflight_sha256':digest(prepath/'manifest.json'),'source_freeze_sha256':'freeze',
              'runner_sha256':digest(resume.ROOT/'scripts/run_source_group_capture_v1.py')}
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'identity':identity}));monkeypatch.setattr(resume,'GATE',gate)
    pre={'sequences':{'seq12':{'source':str(src.parent),'frames':3,'prompt_frame':0,'targets':[1,2],
          'archived_prompt_sha256':'prompt','archived_prompt_mask_sha256':digest(src/'prompt_refined.png')}}}
    ref={'passed':True,'mode':'reference','sequence':'seq12','full_sequence':True,'limit':0,'prompt_frame':0,'targets':[1,2],
         'short_gate_sha256':digest(gate),'prompt_sha256':'prompt','archived_prompt_mask_sha256':digest(src/'prompt_refined.png'),
         'conditioning_sha256':digest(refdir/'conditioning_mask.png'),'observations':[],'identity':identity,
         'label_guard':{'open_attempts':0},'rows':rows,'positions':[],'selection':[],'anchor_events':[],
         'token_window_events':[{'stage':'after_write','total_tokens':1,'insurance_tokens':0} for _ in range(2)]}
    (refdir/'receipt.json').write_text(json.dumps(ref))
    assert resume.validate_reference(run,'seq12',pre,'freeze')['targets']==2
    (refdir/'000001.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='reference mask changed'):resume.validate_reference(run,'seq12',pre,'freeze')


def test_launcher_and_run_manifest_are_rechecked(tmp_path):
    from scripts.memory_token_io import digest
    launcher=tmp_path/'launcher.py';manifest=tmp_path/'run_manifest.json'
    launcher.write_text('source');manifest.write_text('manifest')
    a,b=digest(launcher),digest(manifest)
    resume.verify_launcher_files(launcher,a,manifest,b)
    manifest.write_text('changed')
    with pytest.raises(RuntimeError,match='manifest changed'):
        resume.verify_launcher_files(launcher,a,manifest,b)
    manifest.write_text('manifest');launcher.write_text('changed')
    with pytest.raises(RuntimeError,match='manifest changed'):
        resume.verify_launcher_files(launcher,a,manifest,b)
