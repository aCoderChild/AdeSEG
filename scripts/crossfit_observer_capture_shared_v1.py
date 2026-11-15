"""CPU-only identity and resource checks for the exploratory capture runner."""
import json
import shutil
from pathlib import Path

from PIL import Image
import numpy as np

from datasets.video import load_manifest
from modeling.seq23_viewport import VIEWPORT
from modeling.samurai_auto_prompt import prepare_prompt
from scripts.memory_token_io import digest
from scripts.run_boundary_video_factorial_v1 import BASE,CHECKPOINT,DETECTOR,OUT
from scripts.run_boundary_video_factorial_v1 import PRE
from scripts.source_group_refiner_shared_v2 import dependencies as existing_dependencies
from scripts.preflight_exploratory_crossfit_capture_v2 import DATA,FRESH,EXISTING,uniform_indices

ROOT=Path(__file__).resolve().parents[1]
PLAN_DIR=OUT/'exploratory_crossfit_capture_preflight_v2_attempt01'
PLAN_SHA='44426e0b883e331de59429629e8c9ddaf8471673b71d2e04a44cf7719d152ab2'
GUARD=22*1024**3
MARGIN=4*1024**3
PER_READ=41*1024**2
SHORT=('seq6','seq23')
PROTOCOL=ROOT/'protocols/EXPLORATORY_CROSSFIT_OBSERVER_CAPTURE_V1.md'


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def source_hashes():
    paths=('scripts/crossfit_observer_capture_shared_v1.py',
           'scripts/run_crossfit_observer_capture_v1.py','scripts/run_crossfit_observer_capture_study_v1.py',
           'scripts/audit_crossfit_observer_capture_v1.py','scripts/audit_crossfit_observer_full_sequence_v1.py',
           'modeling/lean_mps_decoder_observer.py',
           'modeling/lean_memory_capture.py','modeling/prompt_feature_refinement.py',
           'scripts/audit_expanded_train_capture_v1.py',
           'modeling/lean_boundary_window_runtime_v1.py','modeling/lean_boundary_refinement_routing_v1.py',
           'modeling/seq23_viewport.py','protocols/EXPLORATORY_CROSSFIT_OBSERVER_CAPTURE_V1.md')
    return {p:digest(ROOT/p) for p in paths}


def validate_plan():
    path=PLAN_DIR/'plan.json';done=read(PLAN_DIR/'completion.json')
    require(digest(path)==PLAN_SHA and done.get('passed') is True and done.get('plan_sha256')==PLAN_SHA,
            'The independently reviewed V2 capture plan changed')
    plan=read(path);require(digest(path)==PLAN_SHA,'Plan changed during parsing')
    require(plan.get('passed') is True and plan.get('capture_authorized') is False and
            plan.get('label_guard',{}).get('open_attempts')==0 and
            plan.get('checkpoint_sha256')==digest(CHECKPOINT) and plan.get('detector_sha256')==digest(DETECTOR) and
            plan.get('data_manifest_sha256')==digest(DATA) and
            plan.get('existing_manifest_sha256')==digest(EXISTING/'manifest.json') and
            plan.get('existing_capture_dependency_sha256')==existing_dependencies() and
            all(digest(ROOT/p)==v for p,v in plan.get('source_sha256',{}).items()), 'Frozen index inputs changed')
    require(len(plan.get('existing_reads',[]))==214 and len(plan.get('fresh_reads',[]))==143 and
            plan.get('source_frames_checked')==2225 and set(plan.get('source_frames',{}))=={f'seq{i}' for i in range(1,24)},
            'Frozen capture coverage differs')
    return plan


def sequence_inputs(plan,sequence):
    require(sequence in FRESH,'Sequence is outside the 143-read plan')
    videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')}
    frames=videos[sequence];records=plan['source_frames'][sequence]
    verify_frames(frames,records)
    prompt=plan['prompts'][sequence]; start=prompt['initial_frame']
    require(type(start) is int and 0<=start<len(frames)-1,'Invalid initialization frame')
    selected=[row['frame'] for row in plan['fresh_reads'] if row['sequence']==sequence]
    require(selected==uniform_indices(start,len(frames)),'Frozen deterministic capture grid differs')
    baseline=Path(prompt['baseline_completion_path']).resolve(strict=True)
    require(baseline.is_relative_to(BASE.resolve()) and digest(baseline)==prompt['baseline_completion_sha256'],
            'Audited baseline completion changed')
    rec=read(baseline);require(digest(baseline)==prompt['baseline_completion_sha256'],'Baseline changed while parsing')
    require(rec.get('passed') is True and rec.get('full_video') is True and rec.get('arm')=='A' and
            rec.get('sequence')==sequence and rec.get('prompt_frame')==start and
            rec.get('label_guard',{}).get('open_attempts')==0 and rec.get('checkpoint_sha256')==digest(CHECKPOINT),
            'Baseline identity differs')
    provenance=prompt['prompt_provenance'];mask_path=Path(provenance['prompt_file']).resolve(strict=True)
    require(mask_path.is_relative_to(BASE.resolve()) and digest(mask_path)==provenance['prompt_mask_sha256']
            and digest(mask_path.parent/'prompt.json')==provenance['prompt_metadata_sha256'],'Frozen automatic prompt changed')
    with Image.open(frames[start].image_path) as image:width,height=image.size
    with Image.open(mask_path) as image:raw_mask=np.asarray(image.convert('L'))
    require(digest(mask_path)==provenance['prompt_mask_sha256'],'Prompt changed during decoding')
    mask,actual_roi=prepare_prompt(raw_mask,sequence,(height,width))
    view={'kind':'fixed_seq23_main_viewport','xyxy':list(VIEWPORT)} if sequence=='seq23' else {'kind':'full_source_frame'}
    roi=list(VIEWPORT) if sequence=='seq23' else [0,0,width,height]
    require(mask.any() and list(actual_roi)==roi and prompt['view']==rec['view']==view and provenance['roi_xyxy']==roi and
            mask.shape==(roi[3]-roi[1],roi[2]-roi[0]),'Prompt geometry or viewport differs')
    return frames,records,start,selected,mask,baseline,rec,prompt


def verify_frames(frames,records):
    require(len(frames)==len(records),'Frame count differs from index plan')
    for i,(frame,row) in enumerate(zip(frames,records,strict=True)):
        require(row['frame']==i and row['source_frame_index']==frame.frame_index and
                row['image_sha256']==digest(frame.image_path),'Source frame identity changed')


def archived_frame_metadata(plan,sequence):
    """Bind historical frame-metadata bytes without deserializing outcome fields."""
    require(sequence in FRESH,'Unknown fresh capture sequence')
    prompt=plan['prompts'][sequence]
    if sequence in ('seq11','seq20','seq23'):
        binding=prompt['source_index_metadata'];path=Path(binding['path']).resolve(strict=True)
        require(binding['split_preflight_sha256']==digest(PRE/'manifest.json') and
                digest(path)==binding['sha256'],'Historical frame metadata changed')
        return {'path':str(path),'sha256':binding['sha256'],'preflight_sha256':binding['split_preflight_sha256']}
    folder=(OUT/'remaining_boundary_video_preflight_v1_attempt04' if sequence in ('seq6','seq10')
            else OUT/'extended_boundary_video_preflight_attempt01')
    path=folder/'manifest.json';done=read(folder/'completion.json');sha=digest(path)
    require(done.get('passed') is True and done.get('manifest_sha256')==sha and
            prompt['prompt_provenance']['preflight_sha256']==sha,'Historical input preflight changed')
    data=read(path);require(digest(path)==sha,'Historical input preflight changed during parsing')
    item=data['sequences'][sequence];meta=Path(item['archive_frames_path']).resolve(strict=True)
    require(meta.is_relative_to(BASE.resolve()) and digest(meta)==item['archive_frames_sha256'],
            'Historical frame metadata changed')
    return {'path':str(meta),'sha256':item['archive_frames_sha256'],'preflight_sha256':sha}


def require_headroom(path,remaining_reads):
    require(type(remaining_reads) is int and 0<=remaining_reads<=143,'Invalid remaining read count')
    free=shutil.disk_usage(path).free;minimum=GUARD+MARGIN+remaining_reads*PER_READ
    require(free>=minimum, f'Insufficient headroom: {free} bytes free, need {minimum} before capture')
    return {'free_bytes':free,'minimum_bytes':minimum,'remaining_reads':remaining_reads}


def phase_artifact(root,name,path):
    """Bind a subprocess artifact to its uniquely completed launch phase."""
    root=Path(root).resolve(strict=True);path=Path(path).resolve(strict=True)
    phases=[json.loads(line) for line in (root/'phases.jsonl').read_text().splitlines() if line.strip()]
    matches=[row for row in phases if row.get('phase')==name]
    require(len(matches)==1,'Missing or duplicate launch phase')
    row=matches[0]
    require(row.get('passed') is True and Path(row.get('artifact_path','')).resolve()==path and
            row.get('artifact_sha256')==digest(path) and
            row.get('log_sha256')==digest(root/(name+'.log')),'Frozen phase artifact changed')
    return row['artifact_sha256']
