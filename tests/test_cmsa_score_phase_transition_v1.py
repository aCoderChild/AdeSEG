"""Real synthetic-label transition after isolated nested audit hooks.

Each scenario runs in a fresh interpreter because CPython audit hooks cannot be
removed. No PolypGen labels, GPU, network, or real experiment artifacts are read.
"""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def fixture_environment(tmp_path):
    manifest=tmp_path/'manifest.jsonl';train=tmp_path/'train.png';target=tmp_path/'target.png'
    from PIL import Image
    import json
    Image.new('L',(1280,720),0).save(train);Image.new('L',(1280,720),0).save(target)
    manifest.write_text('\n'.join(json.dumps({'video_id':seq,'mask':str(path)})
                                  for seq,path in [('seq2',train),('seq23',target)])+'\n')
    site=tmp_path/'sitecustomize.py'
    site.write_text(textwrap.dedent('''
        import os
        import scripts.audit_cmsa_seq23_viewport_predictions_v1 as module
        from scripts.source_group_label_guard_v1 import install_mask_open_guard
        def fake_audit(*args):
            first=install_mask_open_guard(os.environ['SYNTHETIC_MANIFEST'])
            second=install_mask_open_guard(os.environ['SYNTHETIC_MANIFEST'])
            try:
                open(os.environ['SYNTHETIC_TARGET'],'rb')
            except PermissionError:
                blocked=True
            else:
                blocked=False
            return {'passed':True,'nested_hook_blocked':blocked}
        module.audit=fake_audit
    '''))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(tmp_path),str(ROOT)])
    env['SYNTHETIC_MANIFEST']=str(manifest);env['SYNTHETIC_TARGET']=str(target);env['SYNTHETIC_ROOT']=str(tmp_path)
    return env


def run_python(source,env):
    proc=subprocess.run([sys.executable,'-c',textwrap.dedent(source)],cwd=ROOT,env=env,
                        capture_output=True,text=True,timeout=90)
    assert proc.returncode==0,proc.stdout+'\n'+proc.stderr


def test_synthetic_target_is_blocked_before_release(tmp_path):
    env=fixture_environment(tmp_path)
    run_python('''
        import os
        from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
        guard=install_phase_mask_guard(os.environ['SYNTHETIC_MANIFEST'])
        try:
            open(os.environ['SYNTHETIC_TARGET'],'rb')
        except PermissionError:
            pass
        else:
            raise AssertionError('Protected label was readable before verification')
        assert guard['open_attempts']==1
    ''',env)


def test_score_main_reads_synthetic_label_after_isolated_prediction_verification(tmp_path):
    env=fixture_environment(tmp_path)
    run_python('''
        import json,os,sys
        from pathlib import Path
        from PIL import Image
        from scripts.memory_token_io import digest
        import scripts.score_cmsa_seq23_viewport_v1 as score
        root=Path(os.environ['SYNTHETIC_ROOT']);target=Path(os.environ['SYNTHETIC_TARGET'])
        score.DATA=Path(os.environ['SYNTHETIC_MANIFEST']);score.new_output=lambda p:Path(p)
        prediction_audit=root/'fake_prediction_audit.json';prediction_audit.write_text('{}')
        prior=root/'fake_prior.json';sample={'mean_present_dice':None,'present_frames':0,'false_alarms':0,'empty_frames':0}
        prior.write_text(json.dumps({'summaries':{'seq23':{'A':{'post_prompt':sample}}}}));score.PRIOR=prior
        current={'prediction_audit_path':str(prediction_audit.resolve()),'prior_sha256':digest(prior)}
        verification=root/'verification.json';verification.write_text(json.dumps(dict(current,passed=True,
            label_guard={'open_attempts':0},scorer_sha256=digest(Path(score.__file__)))))
        def validate(path):
            assert score.reconstruct_prediction_audit('stage','checkpoint','run')=={'passed':True,'nested_hook_blocked':True}
            return current
        def scored(proof):
            with Image.open(target) as image: assert image.size==(1280,720)
            return []
        score.validate_inputs=validate;score.scored_rows=scored
        score.summary=lambda *args,**kwargs:dict(sample)
        sys.argv=['score','--mode','score','--prediction-audit',str(prediction_audit),
                  '--verification',str(verification),'--output',str(root/'score_out')]
        score.main()
        assert json.loads((root/'score_out/results.json').read_text())['passed']
    ''',env)


def test_independent_metric_audit_reads_synthetic_labels_after_verification(tmp_path):
    env=fixture_environment(tmp_path)
    run_python('''
        import json,os
        from pathlib import Path
        from types import SimpleNamespace
        from PIL import Image
        from scripts.memory_token_io import digest
        import scripts.score_cmsa_seq23_viewport_v1 as score
        import scripts.audit_cmsa_seq23_viewport_scores_v1 as audit
        root=Path(os.environ['SYNTHETIC_ROOT']);target=Path(os.environ['SYNTHETIC_TARGET'])
        manifest=Path(os.environ['SYNTHETIC_MANIFEST']);audit.DATA=manifest
        image=root/'image.png';Image.new('RGB',(1280,720)).save(image)
        predroot=root/'run/predictions/seq23';predroot.mkdir(parents=True)
        first=predroot/'000000.png';Image.new('L',(834,720)).save(first)
        for i in range(1,56):os.link(first,predroot/f'{i:06d}.png')
        predsha=digest(first);targetsha=digest(target);imagesha=digest(image)
        metadata=root/'metadata.json';metadata.write_text(json.dumps([{'label_sha256':targetsha} for _ in range(56)]))
        prior=root/'prior.json';prior.write_text(json.dumps({'rows':{'seq23':[{'A':{'present':False}} for _ in range(56)]},
            'summaries':{'seq23':{'A':{'post_prompt':{}}}}}));audit.PRIOR=prior
        current={'prediction_audit_path':str(root/'fake_prediction_audit.json')}
        proof=dict(current,passed=True,label_guard={'open_attempts':0},scorer_sha256='scorer',
            prediction_audit_sha256='prediction-audit-sha',frame_metadata={'path':str(metadata),'sha256':digest(metadata)},
            prior_sha256=digest(prior),run=str(root/'run'),prediction_sha256=[predsha]*56,images=[imagesha]*56)
        verification=root/'verification.json';verification.write_text(json.dumps(proof))
        scores=root/'scores';scores.mkdir();report=scores/'report.md';report.write_text('synthetic')
        (scores/'results.json').write_text(json.dumps({'passed':True,'verification_sha256':digest(verification),
            'scorer_sha256':'scorer','prediction_audit_sha256':'prediction-audit-sha','report_sha256':digest(report),
            'rows':[{} for _ in range(56)],'summaries':{'full_frame':{},'viewport':{}},
            'all_frame_summaries':{'full_frame':{},'viewport':{}},'prior_project_lean_postprompt':{}}))
        def validate(path):
            assert score.reconstruct_prediction_audit('stage','checkpoint','run')=={'passed':True,'nested_hook_blocked':True}
            return current
        audit.validate_inputs=validate;audit.validate_plan=lambda:{}
        frames=[SimpleNamespace(frame_index=i,image_path=image,mask_path=target) for i in range(56)]
        audit.sequence_inputs=lambda *args:(frames,None,0,None,None,None,None,None)
        audit.metrics=lambda pred,truth:{'present':bool(truth.any())}
        audit.summarize=lambda *args,**kwargs:{};audit.compare=lambda *args,**kwargs:None
        result=audit.audit(verification,scores)
        assert result['passed'] and result['checked_frames']==56 and result['prelabel_guard']['open_attempts']==0
    ''',env)
