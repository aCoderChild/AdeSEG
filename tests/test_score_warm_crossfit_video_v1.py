import numpy as np
import pytest

from scripts import score_warm_crossfit_video_v1 as score
from scripts.audit_boundary_video_factorial_v1 import metrics as independent
from scripts.score_warm_crossfit_v1 import compare


def test_summary_separates_post_prompt_present_and_empty():
    truth = np.zeros((8, 8), bool); truth[2:4, 2:4] = True
    present = score.metric(truth, truth)
    missed = score.metric(np.zeros_like(truth), truth)
    empty = score.metric(np.ones_like(truth), np.zeros_like(truth))
    rows = [{'post_prompt': False, 'A': missed}, {'post_prompt': True, 'A': present},
            {'post_prompt': True, 'A': empty}]
    summary = score.summarise(rows, 'A', True)
    assert summary['frames'] == 2 and summary['present_frames'] == 1
    assert summary['mean_present_dice'] == 1.0 and summary['empty_frames'] == 1 and summary['false_alarms'] == 1
    assert summary['mean_present_boundary_f_5px'] == 1.0 and summary['zero_overlap_present_frames'] == 0
    assert summary['mean_empty_predicted_pixels'] == 64
    assert score.summarise(rows, 'A', False)['mean_present_dice'] == .5
    errors = []; compare(present, independent(truth, truth), [], errors)
    compare(empty, independent(np.ones_like(truth), np.zeros_like(truth)), [], errors)
    assert not errors


def test_failed_video_verification_never_opens_target(tmp_path, monkeypatch):
    monkeypatch.setattr(score, 'install_phase_mask_guard', lambda _: {'open_attempts': 0, 'active': True})
    monkeypatch.setattr(score, 'validate_frozen', lambda *a, **k: (_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(score, 'compute', lambda *a, **k: pytest.fail('target scoring was reached'))
    monkeypatch.setattr(score.sys, 'argv', ['score', '--mode', 'score', '--root', str(tmp_path),
                                            '--output', str(tmp_path / 'score')])
    with pytest.raises(ValueError, match='not frozen'): score.main()
    assert not (tmp_path / 'score').exists()


def test_audit_requires_bound_primary_before_any_target_read(tmp_path, monkeypatch):
    guard = {'open_attempts': 0, 'active': True}
    monkeypatch.setattr(score, 'install_phase_mask_guard', lambda _: guard)
    monkeypatch.setattr(score, 'validate_frozen', lambda *a, **k: {})
    def phase(root, name, path):
        if name == 'verify': return 'vsha'
        raise ValueError('primary score missing')
    monkeypatch.setattr(score, 'phase_artifact', phase)
    monkeypatch.setattr(score, 'read_bound', lambda path, sha: {'passed': True,
        'source_sha256': score.digest(score.Path(score.__file__)), 'label_guard': {'open_attempts': 0}})
    monkeypatch.setattr(score, 'compute', lambda *a, **k: pytest.fail('target scoring was reached'))
    monkeypatch.setattr(score.sys, 'argv', ['score', '--mode', 'audit', '--root', str(tmp_path),
                                            '--output', str(tmp_path / 'audit')])
    with pytest.raises(ValueError, match='primary score missing'): score.main()
    assert guard['active'] and not (tmp_path / 'audit').exists()


def test_report_discloses_denominators_empty_alarms_and_no_init():
    value = {'mean_present_dice': .5, 'false_alarms': 2, 'empty_frames': 3,
             'present_frames': 7}
    result = {'summaries': {'seq23': {a: {'post_prompt': value} for a in score.ARMS}},
        'preprompt': {'seq23': {'present_misses': 4}},
        'overall': {a: {'patient_macro_present_dice': .5, 'patients_with_present_frames': 1,
                        'pooled_post_prompt': value} for a in score.ARMS},
        'paired_vs_A': {a: {'seq23': {'mean_present_dice_difference': 0.}} for a in ('B','C','D')},
        'no_initialization': ['seq1','seq7']}
    result['descriptive_analysis'] = score.descriptive_analysis(result)
    report = score.render_report(result)
    assert '| seq23 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 2/3 / 2/3 / 2/3 / 2/3 | 4 |' in report
    assert 'Seq1 and seq7' in report and 'full canvas' in report and 'not inference-only FPS' in report
    assert 'not the unmodified author Lean baseline' in report
    assert '| seq1 | N/A | N/A | N/A | N/A | N/A | N/A |' in report
    assert '0 / 0 / 1' in report and 'statistical significance or causation' in report


def test_descriptive_analysis_excludes_missing_patient_differences():
    pooled = {'false_alarms': 0, 'empty_frames': 0}
    result = {'overall': {arm: {'pooled_post_prompt': pooled} for arm in score.ARMS},
              'paired_vs_A': {arm: {'seq1': {'mean_present_dice_difference': None}}
                              for arm in ('B', 'C', 'D')}}
    description = score.descriptive_analysis(result)
    assert description['B']['matched_patients'] == 0
    assert description['B']['mean_patient_dice_difference'] is None
    assert description['B']['patients_unchanged'] == 0


def test_video_compute_matches_independent_with_roi_and_preprompt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from scripts.memory_token_io import digest
    root = tmp_path; (root / 'certificate').mkdir(); existing = root / 'existing'; existing.mkdir()
    preflight = root / 'old_preflight.json'; preflight.write_text('{}')
    manifest = existing / 'manifest.json'
    manifest.write_text(json.dumps({'preflight_path': str(preflight), 'preflight_sha256': digest(preflight), 'sequences': {}}))
    monkeypatch.setattr(score, 'EXISTING', existing)
    frames = []
    for i in range(2):
        image = root / f'image{i}.png'; Image.fromarray(np.zeros((720,1280,3), np.uint8)).save(image)
        target = np.zeros((720,1280), np.uint8)
        if i == 1: target[5, 100] = 1; target[5, 500] = 1
        label = root / f'label{i}.png'; Image.fromarray(target).save(label)
        frames.append(SimpleNamespace(image_path=image, mask_path=label, frame_index=i))
    monkeypatch.setattr(score, 'load_manifest', lambda data, split: {'seq23': frames} if split == 'dev' else {})
    monkeypatch.setattr(score, 'validate_plan', lambda: {})
    monkeypatch.setattr(score, 'archived_frame_metadata', lambda *a: {})
    monkeypatch.setattr(score, 'project_label_hashes', lambda binding, count, ids: {i: digest(frames[i].mask_path) for i in ids})
    refdir = root / 'ref'; refdir.mkdir(); refrows = []
    for i in range(2):
        mask = np.zeros((720,1280), np.uint8)
        if i == 1: mask[5,500] = 255
        file = refdir/f'{i:06d}.png'; Image.fromarray(mask).save(file)
        refrows.append({'frame':i, 'mask_sha256':digest(file)})
    ref = refdir/'receipt.json'; ref.write_text(json.dumps({'rows':refrows}))
    spec = {'frames':2, 'prompt_frame':0, 'reference_path':str(ref), 'reference_sha256':digest(ref),
            'image_sha256':[digest(f.image_path) for f in frames], 'roi_xyxy':[446,0,1280,720]}
    cert = root/'certificate/certificate.json'
    cert.write_text(json.dumps({'references':{'seq23':spec}, 'old_manifest_sha256':digest(manifest)}))
    proof = {'certificate_sha256':digest(cert), 'runs':{'seq23':{}}}
    part={'min':-1.0,'max':1.0,'positive_pixels':0,'numel':4,'sha256':'a'*64}
    for arm in ('B','C','D'):
        folder=root/'full'/'seq23'/arm; folder.mkdir(parents=True); masks=[]
        for i in range(2):
            arr=np.zeros((720,1280), np.uint8)
            if i==1 and arm!='C': arr[5,500]=255
            file=folder/f'{i:06d}.png'; Image.fromarray(arr).save(file)
            masks.append({'frame':i,'sha256':digest(file)})
        telemetry=folder/'telemetry.json'; telemetry.write_text(json.dumps([{'frame':1,'conditioning':False,
            'native_object_present':False,'native_exact_suppressed':True,'object_score_logit':-1.,
            'native':part,'refined':part,'output':part,'memory_high_res':part}]))
        rec=folder/'receipt.json'; rec.write_text(json.dumps({'masks':masks,'wall_seconds':1.,'timing_scope':'test',
                                                               'trace_sha256':{'telemetry':digest(telemetry)}}))
        proof['runs']['seq23'][arm]={'path':str(rec),'sha256':digest(rec)}
    primary=score.compute(root,proof); other=score.compute(root,proof,independent=True)
    errors=[]; compare(primary,other,[],errors); assert not errors
    assert primary['summaries']['seq23']['A']['post_prompt']['mean_present_dice'] == pytest.approx(2/3)
    assert primary['seq23_viewport']['A']['post_prompt']['mean_present_dice'] == 1
    assert primary['paired_vs_A']['C']['seq23']['mean_present_dice_difference'] == pytest.approx(-2/3)
    assert primary['overall']['A']['patients_with_present_frames'] == 1
    assert primary['telemetry']['seq23']['C']['native_absent_score_frames'] == 1


def test_all_21_videos_must_freeze_and_tamper_is_rejected(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from PIL import Image
    from scripts import run_warm_crossfit_video_study_v1 as launch
    from scripts.memory_token_io import digest
    monkeypatch.setattr(score, 'OUT', tmp_path)
    monkeypatch.setattr(score, 'sources', lambda: {})
    monkeypatch.setattr(score, 'audit_smoke', lambda *a: {'passed': True})
    videos = {}; refs = {}; checkpoints = {str(f): {'fold': f} for f in range(1, 6)}
    for number in range(1, 24):
        seq=f'seq{number}'
        if seq in ('seq1','seq7'): continue
        shape=(720,1280) if seq=='seq23' else (3,4); fold=(number % 5)+1
        images=[]; refdir=tmp_path/'refs'/seq; refdir.mkdir(parents=True); refrows=[]
        for frame in range(2):
            image=tmp_path/f'{seq}_{frame}.png'; Image.fromarray(np.zeros((*shape,3),np.uint8)).save(image)
            images.append(SimpleNamespace(image_path=image,frame_index=frame))
            saved=refdir/f'{frame:06d}.png'; Image.fromarray(np.zeros(shape,np.uint8)).save(saved)
            refrows.append({'frame':frame,'mask_sha256':digest(saved)})
        videos[seq]=images
        reference=refdir/'receipt.json'; reference.write_text(json.dumps({'rows':refrows,
            'positions':[],'selection':[],'anchor_events':[],'token_window_events':[]}))
        refs[seq]={'frames':2,'fold':fold,'prompt_frame':0,'reference_path':str(reference),
            'reference_sha256':digest(reference),'image_sha256':[digest(x.image_path) for x in images],
            'roi_xyxy':[446,0,1280,720] if seq=='seq23' else None}
    monkeypatch.setattr(score,'load_manifest',lambda data,split:videos if split=='dev' else {})
    crossfit=tmp_path/'crossfit'; preflight=tmp_path/'preflight'
    tasks=launch.phases(tmp_path,crossfit,preflight)
    (tmp_path/'run_manifest.json').write_text(json.dumps({'crossfit_root':str(crossfit),'preflight_path':str(preflight),
        'source_sha256':{},'phases':[x[0] for x in tasks]}))
    cert=tmp_path/'certificate/certificate.json'; cert.parent.mkdir()
    cert.write_text(json.dumps({'passed':True,'source_sha256':{},'label_guard':{'open_attempts':0},
                                'references':refs,'checkpoints':checkpoints}))
    cert_sha=digest(cert)
    smoke_file=tmp_path/'smoke_audit/audit.json'; smoke_file.parent.mkdir()
    smoke_file.write_text(json.dumps({'passed':True,'label_guard':{'open_attempts':0}})); smoke_sha=digest(smoke_file)
    part={'min':-1.0,'max':1.0,'positive_pixels':0,'numel':4,'sha256':'a'*64}
    for seq,spec in refs.items():
        for arm in ('B','C','D'):
            folder=tmp_path/'full'/seq/arm; (folder/'verification').mkdir(parents=True)
            runtime=folder/'verification/runtime_verification.json'
            runtime.write_text(json.dumps({'passed':True,'sequence':seq,'device':'mps','certificate_sha256':cert_sha,
                                           'fold_checkpoint':checkpoints[str(spec['fold'])]}))
            native=[]; masks=[]
            if arm=='B': (folder/'native').mkdir()
            for i,frame in enumerate(videos[seq]):
                with Image.open(frame.image_path) as image: shape=(image.height,image.width)
                file=folder/f'{i:06d}.png'; Image.fromarray(np.zeros(shape,np.uint8)).save(file)
                masks.append({'frame':i,'sha256':digest(file)})
                if arm=='B':
                    n=folder/'native'/f'{i:06d}.png'; Image.fromarray(np.zeros(shape,np.uint8)).save(n)
                    native.append({'frame':i,'sha256':digest(n)})
            trace_sha={}
            for key in ('positions','selection','anchor_events','token_window_events','telemetry'):
                trace=folder/f'{key}.json'
                records=[{'frame':1,'conditioning':False,'native_object_present':False,
                    'native_exact_suppressed':True,'object_score_logit':-1.,'native':part,'refined':part,
                    'output':part,'memory_high_res':part}] if key=='telemetry' else []
                trace.write_text(json.dumps(records)); trace_sha[key]=digest(trace)
            receipt={'passed':True,'sequence':seq,'arm':arm,'limit':0,'certificate_sha256':cert_sha,
                'smoke_audit_sha256':smoke_sha,'reference_sha256':spec['reference_sha256'],'prompt_frame':0,
                'roi_xyxy':spec['roi_xyxy'],'label_guard':{'open_attempts':0},'wall_seconds':1.,'timing_scope':'test',
                'rows':[{'frame':0},{'frame':1}],'masks':masks,'native_masks':native,
                'runtime_verification_sha256':digest(runtime),'trace_sha256':trace_sha}
            (folder/'receipt.json').write_text(json.dumps(receipt))
    phase_rows=[]
    for name,_,artifact in tasks[:-3]:
        if name.startswith('smoke_') and name!='smoke_audit':
            artifact.parent.mkdir(parents=True); artifact.write_text('{}')
        log=tmp_path/f'{name}.log'; log.write_text('done')
        phase_rows.append({'phase':name,'passed':True,'artifact_path':str(artifact),'artifact_sha256':digest(artifact),
                           'log_sha256':digest(log)})
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in phase_rows))
    proof=score.validate_frozen(tmp_path); assert len(proof['references']) == 21
    (tmp_path/'full/seq3/B/000001.png').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='Full mask changed'):
        score.validate_frozen(tmp_path)
