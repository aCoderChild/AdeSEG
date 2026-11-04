"""Audit official DAVIS/SA-V layout and record unchanged author evaluator commands."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.vos_benchmarks import inventory, layout
from scripts.memory_token_io import digest, validate_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['davis2017','sav'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.dataset=='davis2017' and args.split!='val':
        parser.error('This locally scored DAVIS setup uses the 2017 validation split')
    out=validate_output(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    record=inventory(args.root,args.dataset,args.split)
    images,masks,listing,per_object=layout(args.root,args.dataset,args.split)
    prediction=out/'predictions'
    if args.dataset=='davis2017':
        evaluator=ROOT/'external/davis2017-evaluation'
        revision=subprocess.check_output(['git','-C',str(evaluator),'rev-parse','HEAD'],text=True).strip()
        if revision!='ac7c43fca936f9722837b7fbd337d284ba37004b':
            raise RuntimeError('DAVIS evaluator revision differs')
        command=[sys.executable,str(evaluator/'evaluation_method.py'),'--davis_path',str(args.root),
            '--task','semi-supervised','--set','val','--results_path',str(prediction)]
    else:
        revision=record['author_revision']
        command=[sys.executable,str(ROOT/'external/sam2/sav_dataset/sav_evaluator.py'),
            '--gt_root',str(masks),'--pred_root',str(prediction),'--strict','--num_processes','2']
    record.update(root=str(args.root.resolve()),video_list_sha256=digest(listing),
        evaluator_revision=revision,evaluator_command=command,
        inference_status='Not run. Requires dataset-specific multi-object/native replay verification and a frozen evaluated checkpoint.',
        adaptation='No learned confidence gate, tuned threshold, frame-quality selection or temporal subsampling. Native SAM2 mask-logit decoding stays unchanged.',
        sources={n:digest(ROOT/n) for n in ['datasets/vos_benchmarks.py','scripts/prepare_vos_benchmark.py',
            'external/sam2/tools/vos_inference.py','protocols/davis_sav_evaluation_v1.json']})
    (out/'inventory.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f"Audited {record['video_count']} videos / {record['object_count']} objects / {record['input_frames']} input frames. No model inference run.")


if __name__=='__main__':
    main()
