"""DAVIS and SA-V layouts with pinned SAM2 author mask I/O."""
import ast
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from PIL import Image

from modeling.region_tokens import SAM2_REVISION, _verified_source


@lru_cache(maxsize=1)
def author_mask_io():
    path,source=_verified_source('sam2',SAM2_REVISION,'tools/vos_inference.py')
    names={'load_ann_png','save_ann_png','get_per_obj_mask','put_per_obj_mask',
           'load_masks_from_dir','save_masks_to_dir'}
    nodes=[n for n in ast.parse(source).body if
        (isinstance(n,ast.FunctionDef) and n.name in names) or
        (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DAVIS_PALETTE' for t in n.targets))]
    if {n.name for n in nodes if isinstance(n,ast.FunctionDef)}!=names:
        raise RuntimeError('Pinned SAM2 mask I/O differs')
    namespace={'os':os,'np':np,'Image':Image}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def layout(root,dataset,split):
    root=Path(root)
    if dataset=='davis2017':
        return root/'JPEGImages/480p',root/'Annotations/480p',root/f'ImageSets/2017/{split}.txt',False
    if dataset=='sav':
        return root/'JPEGImages_24fps',root/'Annotations_6fps',root/f'sav_{split}.txt',True
    raise ValueError('Expected davis2017 or sav')


def inventory(root,dataset,split):
    images,masks,video_list,per_object=layout(root,dataset,split)
    names=video_list.read_text().splitlines()
    if not names or len(set(names))!=len(names) or any(Path(n).name!=n or n in ('.','..') for n in names):
        raise ValueError('Expected a unique, nonempty official video list')
    tools=author_mask_io();records=[]
    for name in names:
        frames=sorted([p for p in (images/name).iterdir() if p.suffix.lower() in ('.jpg','.jpeg')],key=lambda p:int(p.stem))
        if not frames or len({p.stem for p in frames})!=len(frames):
            raise ValueError('Missing or ambiguous video frames')
        if per_object and [int(p.stem) for p in frames]!=list(range(len(frames))):
            raise ValueError('SA-V 24-fps input must contain every consecutive frame from zero')
        objects,palette=tools['load_masks_from_dir'](str(masks),name,frames[0].stem,per_object)
        if not per_object:
            objects.pop(255,None)  # DAVIS ignored pixels are not an object prompt.
        if not objects or any(not mask.any() for mask in objects.values()):
            raise ValueError('Every declared object needs a nonempty first-frame prompt')
        with Image.open(frames[0]) as image:
            shape=(image.height,image.width)
        if any(mask.shape!=shape for mask in objects.values()):
            raise ValueError('Prompt mask/image geometry differs')
        frame_names={p.stem for p in frames};annotations={}
        if per_object:
            for object_id in objects:
                annotated=sorted(p.stem for p in (masks/name/f'{object_id:03d}').glob('*.png'))
                if len(annotated)<3 or not set(annotated)<=frame_names:
                    raise ValueError('SA-V needs valid first/interior/last annotated frames')
                annotations[str(object_id)]=annotated
        else:
            annotated=sorted(p.stem for p in (masks/name).glob('*.png'))
            if set(annotated)!=frame_names:
                raise ValueError('DAVIS train/val annotations must cover every input frame')
            annotations['indexed']=annotated
        records.append({'video':name,'frames':[p.name for p in frames],
            'object_ids':sorted(objects),'annotations':annotations,'height':shape[0],'width':shape[1]})
    return {'dataset':dataset,'split':split,'videos':records,'video_count':len(records),
        'object_count':sum(len(r['object_ids']) for r in records),
        'input_frames':sum(len(r['frames']) for r in records),
        'per_object_png':per_object,'author_revision':SAM2_REVISION,
        'prompt':'First video frame only; no later annotation guides inference.',
        'evaluation':'Official J&F; first/last annotated frames excluded. Dice is supplementary, computed framewise, not converted from mean J.'}
