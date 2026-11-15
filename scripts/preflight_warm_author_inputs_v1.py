#!/usr/bin/env python3
"""Label-free archived prompt and source-API preflight; no model inference."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.warm_author_initialization_v1 import archived_warm_components
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_study_v2 import CHECKPOINT,DETECTOR,EXTENDED,REMAINING
from scripts.run_additional_author_comparator_v2 import new_output
from scripts.run_author_baseline import source_identity
from scripts.run_author_lean_automatic_v1 import DATA,SEQUENCES,inputs
from scripts.run_native_sam2_automatic_priority import native_source_identity,REPO as NATIVE
from scripts.source_group_label_guard_v1 import install_mask_open_guard

SOURCES=('scripts/preflight_warm_author_inputs_v1.py','modeling/warm_author_initialization_v1.py',
         'modeling/warm_automatic_prompt_v2.py','protocols/WARM_MATCHED_AUTHOR_COMPARATORS_V1.md')


def source_hashes():
    names=set(SOURCES)|{'scripts/local_import_closure_v1.py'}|set(local_import_closure(ROOT,SOURCES))
    return {str(p):digest(ROOT/p) for p in sorted(names)}


def api_evidence():
    repos={'native':(NATIVE,native_source_identity()),'lean':source_identity('lean'),
           'samurai':source_identity('samurai'),'sam2long':source_identity('sam2long')}
    result={}
    for name,(repo,rev) in repos.items():
        path=(repo/'sam2/sam2/sam2_video_predictor.py' if name=='samurai'
              else repo/'sam2/sam2_video_predictor.py')
        tree=ast.parse(path.read_text());classes=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef)]
        candidates=[]
        for cls in classes:
            methods={n.name for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            if {'add_new_points_or_box','add_new_mask','propagate_in_video'}.issubset(methods):
                candidates.append(cls.name)
        if not candidates:raise ValueError(f'{name} lacks the required author API')
        result[name]={'repo':str(repo.resolve()),'revision':rev,'predictor_path':str(path.resolve()),
                      'predictor_sha256':digest(path),'classes':candidates,
                      'propagation':'offline_final_path' if name=='sam2long' else 'forward_video'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=new_output(a.output);guard=install_mask_open_guard(DATA);sources=source_hashes();apis=api_evidence();items={}
    for seq in SEQUENCES:
        frames,start,mask,provenance=inputs(seq,EXTENDED,REMAINING,CHECKPOINT,DETECTOR)
        warm=archived_warm_components(seq,start,mask,provenance)
        items[seq]={'prompt_frame':start,'prompt':provenance,'box':warm['box'].tolist(),
                    'points':warm['points'].tolist(),'point_labels':[1,0],'view_shape':list(mask.shape),
                    'image_sha256':[digest(frame.image_path) for frame in frames]}
    if guard['open_attempts'] or source_hashes()!=sources:raise ValueError('Label opened or source changed')
    result={'passed':True,'scope':'Source API and archived prompt geometry only; no model inference or labels.',
            'sequences':items,'source_sha256':sources,'api_evidence':apis,
            'checkpoint_sha256':digest(CHECKPOINT),'detector_sha256':digest(DETECTOR),
            'label_guard':dict(guard),'limitations':['MPS execution and state behavior remain untested.',
             'SAM2Long uses offline final-path outputs.','Historical point-generation and committed masks need not match.']}
    out.mkdir(parents=True);path=out/'preflight.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequences':len(items),'annotation_opens':guard['open_attempts']}))


if __name__=='__main__':main()
