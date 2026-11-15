"""CPU-only checks for the training/selection-exposed video extension."""
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.preflight_training_exposed_boundary_video_v1 import SEQUENCES, training_exposed_inputs
from scripts.run_training_exposed_boundary_factorial_v1 import ARMS, commands
from scripts.run_training_exposed_boundary_video_short_v1 import commands as short_commands
from scripts import score_training_exposed_boundary_factorial_v1 as scorer
from scripts import audit_training_exposed_boundary_factorial_v1 as auditor


def test_plans_all_36_before_score_and_audit(tmp_path):
    phases,mapping=commands(tmp_path,'preflight','gate','extension','audit')
    assert set(mapping)==set(SEQUENCES)
    assert len({folder for arms in mapping.values() for folder in arms.values()})==36
    assert all(set(arms)==set(ARMS) for arms in mapping.values())
    assert [name for name,_ in phases[-2:]]==['score','independent_audit']
    assert len(phases)==38


def test_short_plan_runs_matched_repeats_before_gate(tmp_path):
    phases,mapping=short_commands(tmp_path,'preflight','gate')
    assert set(mapping)=={'seq2','seq15'}
    assert [name for name,_ in phases[-2:]]==['gate','independent_audit']
    assert len(phases)==12


def test_map_rejects_alias_and_missing_sequences(tmp_path):
    import json
    root=tmp_path/'run';root.mkdir();mapping={}
    for sequence in SEQUENCES:
        mapping[sequence]={}
        for arm in ARMS:
            folder=root/sequence/arm;folder.mkdir(parents=True)
            mapping[sequence][arm]=str(folder)
    path=root/'run_map.json';path.write_text(json.dumps(mapping))
    assert len(scorer.mapped_runs(path,tmp_path))==9
    mapping['seq2']['B']=mapping['seq2']['A'];path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError,match='distinct'):scorer.mapped_runs(path,tmp_path)
    del mapping['seq2'];path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError,match='nine'):scorer.mapped_runs(path,tmp_path)


def test_phase_completion_rejects_changed_receipt(tmp_path):
    from scripts.memory_token_io import digest
    folder=tmp_path/'seq2'/'A';folder.mkdir(parents=True)
    receipt=folder/'completion.json';receipt.write_text('{"passed":true}')
    phases=[{'phase':'seq2_A','completion_sha256':digest(receipt)}]
    mapping={'seq2':{'A':folder}}
    scorer.check_phase_completions(phases,mapping)
    receipt.write_text('{"passed":false}')
    with pytest.raises(ValueError,match='Completion changed'):scorer.check_phase_completions(phases,mapping)


def test_run_validation_uses_phase_receipt_hash_not_new_digest(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import run_training_exposed_boundary_factorial_v1 as launcher
    from scripts.memory_token_io import digest
    monkeypatch.setattr(scorer,'SEQUENCES',('seq2',));monkeypatch.setattr(scorer,'ARMS',('A',))
    study=tmp_path/'study';monkeypatch.setattr(launcher,'STUDY',study)
    for arm in ('region_boundary','mask_only_boundary'):
        p=study/'runs'/arm/'checkpoint.pt';p.parent.mkdir(parents=True);p.write_bytes(b'x')
    image=tmp_path/'image';image.write_bytes(b'image')
    pre=tmp_path/'pre';pre.mkdir();(pre/'manifest.json').write_text(json.dumps({'sequences':{'seq2':{
        'frames':1,'image_sha256':[digest(image)]}}}))
    gate=tmp_path/'gate';gate.write_text(json.dumps({'runtime_identity':{}}))
    folder=tmp_path/'run';folder.mkdir();(folder/'completion.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='differs from launch phase'):
        scorer.verify_runs({'seq2':{'A':folder}},{'seq2':[SimpleNamespace(frame_index=0,image_path=image)]},
                           pre,{'gate':gate,'extension_gate':gate},{('seq2','A'):'old-hash'})


def test_bad_output_or_unfinished_batch_cannot_open_labels(tmp_path,monkeypatch):
    monkeypatch.setattr(scorer,'BASE',tmp_path)
    monkeypatch.setattr(scorer,'validate_output',lambda path:Path(path).resolve())
    monkeypatch.setattr(scorer,'load_label_mask',lambda *_: (_ for _ in ()).throw(AssertionError('label read')))
    monkeypatch.setattr(scorer,'mapped_runs',lambda *_: (_ for _ in ()).throw(ValueError('unfinished')))
    monkeypatch.setattr(scorer.sys,'argv',['score','--experiment-root',str(tmp_path),'--run-map',str(tmp_path/'map'),
                                           '--output',str(tmp_path/'out')])
    with pytest.raises(ValueError,match='unfinished'):scorer.main()


def test_independent_metric_agrees_on_known_masks():
    import numpy as np
    truth=np.zeros((15,15),bool);truth[3:10,3:10]=1
    prediction=np.zeros_like(truth);prediction[4:11,3:10]=1
    actual=auditor.metrics(prediction,truth); expected=scorer.metric(prediction,truth)
    for key in ('dice','iou','precision','recall','boundary_f_5px'):
        assert actual[key]==pytest.approx(expected[key])


def test_scorer_uses_frozen_metadata_and_predictions(tmp_path,monkeypatch):
    import json
    import numpy as np
    from PIL import Image
    from types import SimpleNamespace
    from scripts.memory_token_io import digest
    monkeypatch.setattr(scorer,'SEQUENCES',('seq2',))
    source=tmp_path/'archive'/'seq2';source.mkdir(parents=True)
    images=[]; labels=[]
    for i in range(2):
        image=tmp_path/f'image{i}.png'; label=tmp_path/f'label{i}.png'
        Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(image)
        truth=np.zeros((8,8),dtype=np.uint8)
        if i==1:truth[2:5,2:5]=1
        Image.fromarray(truth).save(label);images.append(image);labels.append(label)
    metadata=source/'frames.json';metadata.write_text(json.dumps([{'label_sha256':digest(x)} for x in labels]))
    pre=tmp_path/'pre';pre.mkdir();(pre/'manifest.json').write_text(json.dumps({'sequences':{'seq2':{
        'source':str(source.parent),'archived_frames_sha256':digest(metadata)}}}))
    mapping={'seq2':{}};receipts={'seq2':{}};frames=[]
    for i in range(2):frames.append(SimpleNamespace(frame_index=i,image_path=images[i],mask_path=labels[i]))
    for arm in ARMS:
        folder=tmp_path/arm;folder.mkdir();records=[]
        for i in range(2):
            mask=np.zeros((8,8),dtype=np.uint8)
            if i==1:mask[2:5,2:5]=255
            file=folder/f'{i:06d}.png';Image.fromarray(mask).save(file)
            records.append({'frame':i,'sha256':digest(file)})
        mapping['seq2'][arm]=folder
        receipts['seq2'][arm]={'prompt_frame':0,'masks':records,
                                'input_images':[{'sha256':digest(x)} for x in images]}
    rows,summaries,_,_=scorer.score(mapping,{'seq2':frames},pre,receipts)
    assert summaries['seq2']['A']['post_prompt']['mean_present_dice']==1
    assert rows['seq2'][1]['B']['dice']==1
    (mapping['seq2']['B']/'000001.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Prediction changed before decode'):
        scorer.score(mapping,{'seq2':frames},pre,receipts)


def test_rejects_unknown_sequence_before_opening_data(tmp_path):
    with pytest.raises(ValueError,match='prior training'):
        training_exposed_inputs('seq23',tmp_path,tmp_path/'model',tmp_path/'detector')


def test_real_seq2_automatic_prompt_without_labels():
    root=Path(__file__).resolve().parents[1]
    pre=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/contrast_refinement_v1/source_group_capture_preflight_attempt03')
    if not pre.exists():pytest.skip('Drive-backed study is not mounted')
    code='''from pathlib import Path
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.run_boundary_video_factorial_v1 import CHECKPOINT,DETECTOR,OUT
from scripts.preflight_training_exposed_boundary_video_v1 import training_exposed_inputs
guard=install_mask_open_guard(Path("data/polypgen_sequence.jsonl"))
frames,prompt,_,view,mask,_=training_exposed_inputs("seq2",OUT/"source_group_capture_preflight_attempt03",CHECKPOINT,DETECTOR)
assert len(frames)==63 and prompt["first_detection_index"]==3 and view=={"kind":"full_source_frame"} and mask.any()
assert guard["open_attempts"]==0
'''
    subprocess.run([sys.executable,'-c',code],cwd=root,check=True)
